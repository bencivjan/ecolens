#include "ffdec.h"
#include <iostream>

ffdec::ffdec()
{
	this->avFrame = NULL;
	this->avCodecContext =  NULL;
	this->avCodec = NULL;
	this->buf = NULL;
	Init();
}

void ffdec::yuv2RGB(int width, int height, const void** src, int* src_linesize, void** dst) {
	struct SwsContext *img_convert_ctx = sws_getContext(
		width, height, AV_PIX_FMT_YUV420P,
		width, height, AV_PIX_FMT_RGB24,
		SWS_FAST_BILINEAR, NULL, NULL, NULL);
	int dstwidth[] = { width * 3, 0, 0 };
	if (img_convert_ctx) {
		sws_scale(img_convert_ctx, (const uint8_t *const*)src, src_linesize, 0, height,
			(uint8_t *const*)dst, dstwidth);
	}
	sws_freeContext(img_convert_ctx);
}

bool ffdec::Init() {
    //if it has been initialized before, we should do clean_up first
    clean_up();

    avformat_network_init();
    avCodec = avcodec_find_decoder(AV_CODEC_ID_H264);
    if (!avCodec) {
        std::cerr << "Decoder not found!!" << std::endl;
        return false;
    }

    avCodecContext = avcodec_alloc_context3(avCodec);
    if (!avCodecContext) {
        //failed to allocate codec context
        clean_up();
        return false;
    }

    if (avcodec_open2(avCodecContext, const_cast<AVCodec*>(avCodec), NULL) < 0) {
        //failed to open codec
        clean_up();
        return false;
    }

    if (avCodecContext->codec_id == AV_CODEC_ID_H264) {
        avCodecContext->flags2 |= AV_CODEC_FLAG2_CHUNKS;
    }
    avFrame = av_frame_alloc();
    if (!avFrame) {
        //failed to allocate frame
        clean_up();
        return false;
    }
    return true;
}

ffdec::~ffdec(void)
{
	clean_up();
	if (buf) {
		delete[] buf;
	}
}

void ffdec::clean_up() {
	if (avFrame != NULL) {
		av_frame_free(&avFrame);
		avFrame = NULL;
	}
	if (avCodecContext != NULL) {
		avcodec_close(avCodecContext);
		avcodec_free_context(&avCodecContext);
		avCodecContext = NULL;
	}
}

pybind11::array_t<uint8_t> ffdec::process_frame(pybind11::array_t<uint8_t> frame) {
	AVPacket avpkt = { 0 };
	av_init_packet(&avpkt);
	auto frame_buf = frame.request();
	avpkt.data = (uint8_t*)frame_buf.ptr;
	avpkt.size = frame_buf.size;
	int got_frame = 0;

	if (avcodec_send_packet(avCodecContext, &avpkt) < 0) {
		std::cerr << "Error sending a packet for decoding" << std::endl;
		return pybind11::array_t<char>(0);
	}

	int len = avcodec_receive_frame(avCodecContext, avFrame);
	if (len < 0) {
		std::cerr << "Error decoding frame" << std::endl;
		return pybind11::array_t<char>(0);
	}

	// yuv2rgb()
	if (buf == NULL) {
		buf = new char[avFrame->width * avFrame->height * 3];
	}
	char* dst[] = { buf, 0, 0 };
	yuv2RGB(avFrame->width, avFrame->height, (const void**)avFrame->data, (int*)avFrame->linesize, (void**)dst);
	auto result = pybind11::array_t<char>(avFrame->width * avFrame->height * 3);
	result.resize({ avFrame->height, avFrame->width, 3 });
	auto ptr = static_cast<char*>(result.request().ptr);
	memcpy(ptr, dst[0], avFrame->width * avFrame->height * 3);
	return result;
}

PYBIND11_MODULE(ffdec, m) {
    pybind11::class_<ffdec>(m, "ffdec")
        .def(pybind11::init<>())
        .def("process_frame", &ffdec::process_frame);
}