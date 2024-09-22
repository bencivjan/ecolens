export PKG_CONFIG_PATH=$PWD/x264/lib/pkgconfig:$PKG_CONFIG_PATH

echo "Installing on $(uname)"

if [ "$(uname)" = "Darwin" ]; then
  DARWIN=1
elif [ "$(uname)" = "Linux" ]; then
  LINUX=1
else
  echo "OS may be unsupported!"
fi

if [ ! -d "x264" ]; then
  git clone https://code.videolan.org/videolan/x264.git --depth=1 x264
fi
cd x264
./configure --enable-static --disable-opencl --prefix=./ --enable-pic
make -j
make install
cd ..

if [ ! -d "ffmpeg" ]; then
  git clone https://git.ffmpeg.org/ffmpeg.git --depth=1 ffmpeg
fi
cd ffmpeg
CFLAGS="-O3 -fPIC" ./configure --prefix=./ \
  --enable-pic \
  --enable-static \
  --enable-small \
  --disable-everything \
  --enable-avcodec \
  --enable-avformat \
  --enable-swscale \
  --enable-avutil \
  --enable-libx264 \
  --enable-decoder=h264 \
  --enable-encoder=libx264 \
  --pkg-config-flags="--static" \
  --extra-cflags="-I../x264/include" \
  --extra-ldflags="-L../x264/lib" \
  --enable-gpl 
make -j
make install
cd ..

c++ -O3 -Wall -shared -std=c++11 ${LINUX:+-fPIC} \
    ${DARWIN:+-undefined dynamic_lookup} \
    -Wl,${LINUX:+-Bsymbolic} \
    -Iffmpeg/include/ \
    -Ix264/include/ \
    $(python3 -m pybind11 --includes) \
    ffenc/ffenc.cc -o ffenc$(python3-config --extension-suffix) \
    -Lffmpeg/lib/ \
    -Lx264/lib/ \
    ${DARWIN:+-L/opt/homebrew/lib} \
    -lavformat -lavcodec -lavutil -lswscale \
    -lx264 \
    -lm -lpthread -lz -ldl -lX11 \
    -lstdc++ -lc ${LINUX:+-lrt} -lvdpau ${LINUX:+-ldrm} \

c++ -O3 -Wall -shared -std=c++11 ${LINUX:+-fPIC} \
    ${DARWIN:+-undefined dynamic_lookup} \
    -Wl,${LINUX:+-Bsymbolic} \
    -Iffmpeg/include/ \
    -Ix264/include/ \
    $(python3 -m pybind11 --includes) \
    ffdec/ffdec.cc -o ffdec$(python3-config --extension-suffix) \
    -Lffmpeg/lib/ \
    -Lx264/lib/ \
    ${DARWIN:+-L/opt/homebrew/lib} \
    ${DARWIN:+-framework CoreVideo} \
    -lavformat -lavcodec -lavutil -lswscale \
    -lx264 \
    -lm -lpthread -lz -ldl -lX11 \
    -lstdc++ -lc ${LINUX:+-lrt} -lvdpau ${LINUX:+-ldrm} \

echo "Done!"