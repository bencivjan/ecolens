from setuptools import setup, Extension

functions_module = Extension(
    name='ffdec',
    sources=['ffdec.cc'],
    extra_compile_args=["-O3", "-fPIC", "-std=c++11", "-Wall"],
    include_dirs=[
        '/home/ryan/anaconda3/envs/py38/include/python3.8',
        '/home/ryan/anaconda3/envs/py38/lib/python3.8/site-packages/pybind11/include',
        '/home/ryan/ffenc/ffmpeg/include',
        '/home/ryan/ffenc/x264/include'
    ],
    library_dirs=[
        '/home/ryan/anaconda3/envs/py38/lib',
        '/home/ryan/ffenc/ffmpeg/lib',
        '/home/ryan/ffenc/x264/lib'
    ],
    libraries=['avcodec', 'avformat', 'avutil', 'swscale', 'x264', 'm', 'pthread', 'z', 'dl', 'X11', 'stdc++', 'c', 'rt', 'vdpau'],
    extra_link_args=["-Wl,-Bsymbolic"],
)

setup(ext_modules=[functions_module])
