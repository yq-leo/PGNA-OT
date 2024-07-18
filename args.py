from argparse import ArgumentParser


def make_args():
    parser = ArgumentParser()
    parser.add_argument('--dataset', dest='dataset', type=str, default='ACM-DBLP',
                        choices=['ACM-DBLP', 'cora', 'foursquare-twitter', 'phone-email', 'Douban', 'flickr-lastfm'],
                        help='datasets: ACM-DBLP; cora; foursquare-twitter; phone-email; Douban; flickr-lastfm')
    parser.add_argument('--ratio', dest='ratio', type=float, default=0.2,
                        choices=[0.2], help='training ratio: 0.2')
    parser.add_argument('--use_attr', dest='use_attr', default=False, action='store_true',
                        help='use input node attributes')

    # Device settings
    parser.add_argument('--gpu', dest='device', action='store_const', const='cuda', default='cpu', help='use GPU')

    # Model settings
    parser.add_argument('--model', dest='model', type=str, default='BRIGHT')
    parser.add_argument('--num_layers', dest='num_layers', type=int, default=1, help='number of layers')
    parser.add_argument('--hidden_dim', dest='hidden_dim', type=int, default=128, help='hidden dimension')
    parser.add_argument('--out_dim', dest='out_dim', type=int, default=128, help='output dimension')

    # Loss settings
    parser.add_argument('--lambda_edge', dest='lambda_edge', type=float, default=35, help='GW weight')
    parser.add_argument('--lambda_total', dest='lambda_total', type=float, default=1e-2, help='weight of entropy regularization')
    parser.add_argument('--in_iter', dest='in_iter', type=int, default=5, help='number of inner iterations')
    parser.add_argument('--out_iter', dest='out_iter', type=int, default=20, help='number of outer iterations')

    # Training settings
    parser.add_argument('--lr', dest='lr', type=float, default=1e-3, help='learning_rate')
    parser.add_argument('--epochs', dest='epochs', type=int, default=250, help='number of epochs')
    parser.add_argument('--runs', dest='runs', type=int, default=1, help='number of runs')

    args = parser.parse_args()
    return args
