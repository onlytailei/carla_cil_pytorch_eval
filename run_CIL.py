import argparse
import logging
import sys

try:
    sys.path.append("../")
    # from carla import carla_server_pb2 as carla_protocol
    from carla.driving_benchmark import run_driving_benchmark
    # from carla.driving_benchmark.experiment_suites import CoRL2017

    from agents.imitation.imitation_learning_pytorch import ImitationLearning
    from benchmarks.vrg_transfer import VrgTransferSuite
except ImportError:
    raise RuntimeError(
        'cannot import "carla_server_pb2.py", run the protobuf compiler to generate this file')

if (__name__ == '__main__'):
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument(
        '-v', '--verbose',
        action='store_true',
        dest='debug',
        help='print debug information')
    argparser.add_argument(
        '--host',
        metavar='H',
        default='localhost',
        help='IP of the host server (default: localhost)')
    argparser.add_argument(
        '-p', '--port',
        metavar='P',
        default=2000,
        type=int,
        help='TCP port to listen to (default: 2000)')
    argparser.add_argument(
        '-c', '--city-name',
        metavar='C',
        default='Town01',
        help='The town that is going to be used on benchmark'
             + '(needs to match active town in server, options: Town01 or Town02)')
    argparser.add_argument(
        '-n', '--log_name',
        metavar='T',
        default='test',
        help='The name of the log file to be created by the scripts'
    )

    argparser.add_argument(
        '--avoid-stopping',
        default=True,
        action='store_false',
        help=' Uses the speed prediction branch to avoid unwanted agent stops'
    )
    argparser.add_argument(
        '--continue-experiment',
        action='store_true',
        help='If you want to continue the experiment with the given log name'
    )
    argparser.add_argument(
        '--model-path',
        metavar='P',
        default='/model/policy.pth',
        type=str,
        help='torch imitation learning model path (relative in model dir)'
    )
    argparser.add_argument(
        '--vrg-transfer',
        default=False,
        action='store_true',
        help='If you want to enable model transfer'
    )
    argparser.add_argument(
        '--vrg-model-path',
        default='/model/transfer.pth',
        type=str,
        help='vrg transfer model path (relative in model dir)'
    )
    argparser.add_argument(
        '--visualize',
        default=False,
        action='store_true',
        help='visualize the image and transfered image through tensorflow'
    )
    argparser.add_argument(
        '--weathers',
        nargs='+',
        type=int,
        default=[1],
        help='weather list 1:clear 3:wet, 6:rain 8:sunset'
    )

    args = argparser.parse_args()

    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(format='%(levelname)s: %(message)s', level=log_level)

    logging.info('listening to server %s:%s', args.host, args.port)

    agent = ImitationLearning(args.city_name,
                              args.avoid_stopping,
                              args.model_path,
                              args.vrg_transfer,
                              args.vrg_model_path,
                              args.visualize
                              )

    # experiment_suites = CoRL2017(args.city_name)
    experiment_suites = VrgTransferSuite(args.city_name, args.weathers)

    # Now actually run the driving_benchmark
    run_driving_benchmark(agent, experiment_suites, args.city_name,
                          args.log_name, args.continue_experiment,
                          args.host, args.port)
