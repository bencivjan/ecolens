#pragma once
extern "C"{
#ifndef INT64_C
#define INT64_C(c) (c ## LL)
#define UINT64_C(c) (c ## ULL)
#endif
#include "libavformat/avformat.h"
#include "libavcodec/avcodec.h"
#include "libswscale/swscale.h"
#include "libavutil/avutil.h"
#include "libavutil/log.h"
#include "libavutil/opt.h"       // for av_opt_set  
}
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

class ffenc
{
public:
	ffenc(int width, int height, int fps);
	~ffenc(void);
	void clean_up();
	pybind11::array_t<uint8_t> process_frame(pybind11::array_t<uint8_t> raw_pic);
	void change_settings(int bitrate_value, int fps);
protected:
	const AVCodec *video_codec;
	AVCodecContext *c;
	AVFrame *frame;
	AVPacket avpkt;
	int mwidth;
	int mheight;
	int mfps;
private:
	bool init();
	char *yy, *uu, *vv;
	static void RGB2yuv(int width, int height, const void* src, void** dst);
};
