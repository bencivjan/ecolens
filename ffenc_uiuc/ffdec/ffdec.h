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
#include "libavutil/opt.h"       // for av_opt_set  
}
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

class ffdec
{
public:
	ffdec();
	~ffdec(void);
	void clean_up();
	pybind11::array_t<uint8_t> process_frame(pybind11::array_t<uint8_t> raw_pic);
protected:
	const AVCodec *avCodec;
	AVCodecContext *avCodecContext;
	AVFrame *avFrame;
	AVPacket avpkt;
private:
	bool Init();
	char *buf;
	static void yuv2RGB(int width,int height,const void** src,int* src_linesize,void** dst);
};