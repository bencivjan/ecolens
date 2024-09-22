#include "ffenc.h"
#include <iostream>

ffenc::ffenc(int width, int height, int fps)
{
	mwidth = width;
	mheight = height;
	mfps = fps;
	video_codec = NULL;
	this->c = NULL;
	this->frame = NULL;
	init();
}

void ffenc::RGB2yuv(int width, int height, const void *src, void **dst)
{
	struct SwsContext *img_convert_ctx = sws_getContext(
		width, height, AV_PIX_FMT_RGB24,
		width, height, AV_PIX_FMT_YUV420P,
		SWS_FAST_BILINEAR, NULL, NULL, NULL);
	uint8_t *rgb_src[3] = {(uint8_t *)src, NULL, NULL};

	int srcwidth[] = {width * 3, 0, 0};
	int dstwidth[] = {width, width / 2, width / 2};
	if (img_convert_ctx)
	{
		sws_scale(img_convert_ctx, (const uint8_t *const *)rgb_src, srcwidth, 0, height,
				  (uint8_t *const *)dst, dstwidth);
	}
	sws_freeContext(img_convert_ctx);
	return;
}

bool ffenc::init()
{
	av_log_set_level(AV_LOG_QUIET);
	clean_up();

	video_codec = avcodec_find_encoder(AV_CODEC_ID_H264);
	if (!video_codec) {
		clean_up();
		return false;
	}
	c = avcodec_alloc_context3(video_codec);
	AVDictionary *opts = NULL;
	c->width = mwidth;
	c->height = mheight;
	c->gop_size = 1000000;
	AVRational ration = {1, mfps};
	c->time_base = ration;
	c->pix_fmt = AV_PIX_FMT_YUV420P;
	c->max_b_frames = 0;
	
	// int br = 500 * 1000;
	// int br = 50000 * 1000;
	int br = 25000 * 1000;
	c->bit_rate = br;
	c->rc_min_rate = br;
	c->rc_max_rate = br;
	c->rc_buffer_size = br / mfps;
	c->rc_initial_buffer_occupancy = c->rc_buffer_size;
	// c->rc_buffer_aggressivity = (float)1.0;
	// c->rc_initial_cplx = 0.5;
	av_opt_set(c->priv_data, "preset", "ultrafast", 0); //ultrafast,superfast, veryfast, faster, fast, medium, slow, slower, veryslow,placebo.
	// av_opt_set(c->priv_data, "profile", "baseline", 0); //baseline main high
	av_opt_set(c->priv_data, "level", "4.2", 0);
	av_opt_set(c->priv_data, "tune", "zerolatency", 0); //  tune

	/* open the codec */
	int ret = avcodec_open2(c, video_codec, &opts);
	if (ret < 0) {
		clean_up();
		return false;
	}

	/* allocate and init a re-usable frame */
	frame = av_frame_alloc();
	if (!frame) {
		clean_up();
		return false;
	}
	frame->format = c->pix_fmt;
	frame->width = c->width;
	frame->height = c->height;
	

	yy = new char[mwidth * mheight];
	uu = new char[mwidth * mheight / 4];
	vv = new char[mwidth * mheight / 4];

	return true;
}

ffenc::~ffenc(void)
{
	clean_up();
	if (yy){
		delete[] yy;
		delete[] uu;
		delete[] vv;
	}
}

void ffenc::change_settings(int bitrate_kbps, int fps)
{
    int value = bitrate_kbps * 1000;
	if (fps <= 0) {
		fps = 1;
	}
	c->bit_rate = value;
	c->rc_min_rate = value;
	c->rc_max_rate = value;
	c->rc_buffer_size = value / fps;
	c->rc_initial_buffer_occupancy = c->rc_buffer_size;
	// c->gop_size = fps;
	AVRational ration = {1, fps};
	c->time_base = ration;
}

void ffenc::clean_up()
{
	if (frame) {
		av_free(frame);
		frame = NULL;
	}
	if (c){
		avcodec_close(c);
		av_free(c);
		c = NULL;
	}
}

pybind11::array_t<uint8_t> ffenc::process_frame(pybind11::array_t<uint8_t> raw_pic) {
    auto src_buf = raw_pic.request();
    char* tmpsrc[] = {0, 0, 0}; 
    char* tmpsrc2[] = {0, 0, 0};
    int tmplinesize[] = {mwidth, mwidth / 2, mwidth / 2 };
    tmpsrc[0] = yy; tmpsrc[1] = uu; tmpsrc[2] = vv;
    tmpsrc2[0] = yy; tmpsrc2[1] = vv; tmpsrc2[2] = uu;
    RGB2yuv(mwidth, mheight, src_buf.ptr, (void**) tmpsrc);

    for (int i = 0; i < 3; i++) {
        frame->data[i] = (unsigned char*) tmpsrc2[i];
        frame->linesize[i] = tmplinesize[i];
    }

    if (avcodec_send_frame(c, frame) < 0) {
        return pybind11::array_t<uint8_t>(0);
    }

	AVPacket* pkt = av_packet_alloc(); 
	if (!pkt) {
		return pybind11::array_t<uint8_t>(0);
	}

	if (avcodec_receive_packet(c, pkt) == 0) {
		auto result = pybind11::array_t<uint8_t>(pkt->size);
		auto ptr = static_cast<uint8_t *>(result.request().ptr);
		memcpy(ptr, pkt->data, pkt->size);
		av_packet_unref(pkt);
		av_packet_free(&pkt);
		return result;
	} else {
		av_packet_free(&pkt);
		return pybind11::array_t<uint8_t>(0);
	}

}


PYBIND11_MODULE(ffenc, m) {
    pybind11::class_<ffenc>(m, "ffenc")
        .def(pybind11::init<int, int, int>())
        .def("process_frame", &ffenc::process_frame)
        .def("change_settings", &ffenc::change_settings);
}