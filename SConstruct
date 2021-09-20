import os

VariantDir('build', 'src', duplicate=0)

AddOption('--d', dest='debug', default=False, action='store_true', help='Debug')

env = Environment(ENV = os.environ)
env.Append(BUILD_BASE_DIR = Dir('.').abspath)

TIMELOOP_PATH = ARGUMENTS.get('timeloop_path', None)
if TIMELOOP_PATH != None:
    env.Append(TIMELOOP_BASE_DIR = Dir(TIMELOOP_PATH).abspath)
else:
    print("ERROR: Please specify Timeloop repository path.")
    Exit(2)

env.SConscript('build/SConscript', exports='env')
