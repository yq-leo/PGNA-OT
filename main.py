import os.path

from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import CosineAnnealingLR
from collections import defaultdict
import time

from args import *
from utils import *
from model import *


if __name__ == '__main__':
    args = make_args()

    # check compatibility between dataset and use_attr
    if args.dataset == 'noisy-cora1-cora2':
        assert args.use_attr is True, 'noisy-cora1-cora2 requires using node attributes'
    elif args.dataset == 'foursquare-twitter' or args.dataset == 'phone-email':
        assert args.use_attr is False, f'{args.dataset} does not have node attributes'

    # load data and build networkx graphs
    np_dtype = np.float64
    torch_dtype = torch.float64
    print("Loading data...", end=" ")
    edge_index1, edge_index2, x1, x2, anchor_links, test_pairs = load_data(f"datasets/{args.dataset}", args.ratio,
                                                                           args.use_attr, dtype=np_dtype)
    anchor1, anchor2 = anchor_links[:, 0], anchor_links[:, 1]
    G1, G2 = build_nx_graph(edge_index1, anchor1, x1), build_nx_graph(edge_index2, anchor2, x2)
    print("Done")

    rwr1, rwr2 = get_rwr_matrix(G1, G2, anchor_links, args.dataset, args.ratio, dtype=np_dtype)
    if x1 is None:
        x1 = rwr1
    if x2 is None:
        x2 = rwr2

    # device setting
    assert torch.cuda.is_available() or args.device == 'cpu', 'CUDA is not available'
    device = torch.device(args.device)
    torch.set_default_dtype(torch.float64)

    # build PyG Data objects
    G1_tg = build_tg_graph(edge_index1, x1, rwr1, dtype=torch_dtype).to(device)
    G2_tg = build_tg_graph(edge_index2, x2, rwr2, dtype=torch_dtype).to(device)

    # model setting
    input_dim = G1_tg.x.shape[1]
    hidden_dim = args.hidden_dim
    output_dim = args.out_dim

    if not os.path.exists('logs'):
        os.makedirs('logs')
    writer = SummaryWriter(save_path(args.dataset, 'logs', args.use_attr))

    max_hits_list = defaultdict(list)
    max_mrr_list = []
    emb_list = []
    for run in range(args.runs):
        print(f"Run {run + 1}/{args.runs}")

        # model = RWRNet(3, input_dim, output_dim).to(device)
        model = BRIGHT(input_dim=input_dim, hidden_dim=output_dim, output_dim=output_dim).to(device)
        if args.model == 'PGNA':
            model = PGNA(input_dim=input_dim,
                         feature_dim=output_dim,
                         anchor_dim=anchor_links.shape[0],
                         hidden_dim=output_dim,
                         output_dim=output_dim,
                         num_layers=args.num_layers).to(device)
        elif args.model == 'RWRNet':
            model = RWRNet(num_layers=args.num_layers,
                           input_dim=input_dim,
                           output_dim=output_dim).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        criterion = FusedGWLoss(G1_tg, G2_tg, anchor1, anchor2,
                                lambda_w=args.lambda_w,
                                lambda_edge=args.lambda_edge,
                                lambda_total=args.lambda_total,
                                in_iter=args.in_iter,
                                out_iter=args.out_iter,
                                total_epochs=args.epochs).to(device)
        scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

        print("Training...")
        max_hits = defaultdict(int)
        max_mrr = 0
        for epoch in range(args.epochs):
            model.train()
            start = time.time()
            optimizer.zero_grad()
            out1, out2 = model(G1_tg, G2_tg)
            loss = criterion(out1=out1, out2=out2, epoch=epoch)
            loss.backward()
            optimizer.step()
            print(f'Epoch {epoch + 1}, Loss: {loss.item():.6f}', end=', ')

            # testing
            with torch.no_grad():
                model.eval()
                inter_c = torch.exp(-(out1 @ out2.T))
                intra_c1 = torch.exp(-(out1 @ out1.T)) * G1_tg.adj
                intra_c2 = torch.exp(-(out2 @ out2.T)) * G2_tg.adj
                similarity = sinkhorn_stable(inter_c, intra_c1, intra_c2,
                                      lambda_w=args.lambda_w,
                                      lambda_e=args.lambda_edge,
                                      lambda_t=args.lambda_total,
                                      in_iter=args.in_iter,
                                      out_iter=args.out_iter,
                                      device=device)
                hits, mrr = compute_metrics(-similarity, test_pairs)
                cost = inter_c / inter_c.sum()
                cost_entropy = torch.sum(-cost * torch.log(cost))
                s_entropy = torch.sum(-similarity * torch.log(similarity))
                end = time.time()
                print(f'cost_entropy: {cost_entropy:.6f}, s_entropy: {s_entropy:.6f}, '
                      f'{", ".join([f"Hits@{key}: {value:.4f}" for (key, value) in hits.items()])}, MRR: {mrr:.4f}')

                max_mrr = max(max_mrr, mrr.cpu())
                for key, value in hits.items():
                    max_hits[key] = max(max_hits[key], value.cpu())

                writer.add_scalar('Loss', loss.item(), epoch)
                writer.add_scalar('MRR', mrr, epoch)
                for key, value in hits.items():
                    writer.add_scalar(f'Hits/Hits@{key}', value, epoch)
                interc_img = torch.exp(-(out1 @ out2.T))
                interc_img /= interc_img.sum()
                interc_img = interc_img / interc_img.max()
                interc_img = interc_img.repeat(3, 1, 1)
                sim_img = similarity / similarity.max()
                sim_img = sim_img.repeat(3, 1, 1)
                writer.add_image('InterC', interc_img, epoch)
                writer.add_image('Similarity', sim_img, epoch)

            # scheduler.step()

        emb_list.append({'out1': out1.detach().cpu().numpy(), 'out2': out2.detach().cpu().numpy()})

        for key, value in max_hits.items():
            max_hits_list[key].append(value)
        max_mrr_list.append(max_mrr)

        print("")

    max_hits = {}
    for key, value in max_hits_list.items():
        max_hits[key] = np.array([val.cpu() for val in value]).mean()
    max_mrr = np.array(max_mrr_list).mean()

    idx = np.argmax(np.array(max_mrr_list))
    out1, out2 = emb_list[idx]['out1'], emb_list[idx]['out2']

    hparam_dict = {
        'dataset': args.dataset,
        'use_attr': args.use_attr,
        'model': args.model,
        'epochs': args.epochs,
        'lr': args.lr,
        'lambda_edge': args.lambda_edge,
        'lambda_total': args.lambda_total,
        'lambda_w': args.lambda_w
    }
    writer.add_hparams(hparam_dict,
                       {'hparam/MRR': max_mrr, **{f'hparam/Hits@{key}': value for key, value in max_hits.items()}})

    if args.save_outputs:
        output_path = save_path(args.dataset, 'outputs', args.use_attr)
        os.makedirs(output_path)
        np.savez(f"{output_path}/pgna_embeddings.npz", out1=out1, out2=out2)
