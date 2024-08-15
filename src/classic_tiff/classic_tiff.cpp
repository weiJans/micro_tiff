#include "classic_tiff.h"
#include "..\micro_tiff\micro_tiff.h"
#include "..\lzw\lzw.h"
#include "..\lzw\data_predict.h"

#include <omp.h>
#include <io.h>
#include <fcntl.h>
#include <stdlib.h>
#include <cmath>
#include <cstring>
#include <direct.h>
#include <functional>
#include <memory>
#include <algorithm>

#define OMP_COMPRESS_TILE_HEIGHT 32

using namespace std;
using namespace tiff;

int32_t save_with_zlib(int32_t hdl, uint32_t ifd_no, void* buf, ImageInfo info)
{
	int32_t status = ErrorCode::STATUS_OK;
	return status;
}

int32_t save_with_lzw_horidif(int32_t hdl, uint32_t ifd_no, void* buf, ImageInfo info)
{
	int32_t status = ErrorCode::STATUS_OK;

	uint32_t block_pixel_width = info.image_width * info.samples_per_pixel;		//RGB(3), GRAY(1)
	uint32_t block_stride = info.image_byte_count * block_pixel_width;

	int32_t strip_count = (int32_t)ceil((double)info.image_height / info.block_height);
	int32_t threads = (min)(omp_get_num_procs() / 2, strip_count);
	threads = (max)(1, threads);

	size_t src_strip_size = info.block_height * (size_t)block_stride;
	//sometimes LZW compress will make data more large, so alloc more buffer
	size_t strip_alloc_size = (size_t)(src_strip_size * 1.5);
	uint8_t* total_buffer = (uint8_t*)calloc(threads, strip_alloc_size);
	if (total_buffer == nullptr) {
		return ErrorCode::ERR_BUFFER_IS_NULL;
	}

	if (!omp_in_parallel())
		omp_set_num_threads(threads);

#pragma omp parallel for
	for (int i = 0; i < strip_count; i++)
	{
		if (status == ErrorCode::STATUS_OK)
		{
			uint32_t strip_height = info.block_height;
			if ((i + 1) * info.block_height > info.image_height) {
				strip_height = (info.image_height - i * info.block_height);
			}

			uint64_t src_len = (uint64_t)strip_height * block_stride;
			uint8_t* src_buf = (uint8_t*)buf + i * src_strip_size;

			int hori_status = horizontal_differencing(src_buf, strip_height, info.image_width, info.image_byte_count, info.samples_per_pixel, false);
			if (hori_status != 0) {
				status = ErrorCode::ERR_LZW_HORIZONTAL_DIFFERENCING;
				continue;
			}

			int thread_num = omp_get_thread_num();

			uint8_t* dst_buf = total_buffer + thread_num * strip_alloc_size;
			uint64_t raw_data_used_size, dst_len;
			int encode_status = LZWEncode(src_buf, src_len, &raw_data_used_size, dst_buf, strip_alloc_size, &dst_len);
			if (encode_status != 1 || raw_data_used_size != src_len) {
				status = ErrorCode::ERR_COMPRESS_LZW_ERROR;
			}
			status = micro_tiff_SaveBlock(hdl, (uint16_t)ifd_no, i, dst_len, dst_buf);
			if (status != 0) {
				continue;
			}
			//make source buffer restore to original state
			status = horizontal_acc(src_buf, strip_height, info.image_width, info.image_byte_count, info.samples_per_pixel, false);
		}
	}

	free(total_buffer);
	return status;
}

int32_t info_conversion(tiff::SingleImageInfo& image_info, ImageInfo& info)
{
	uint16_t tiffCompression;
	uint16_t tiffPredictor = PREDICTOR_NONE;
	uint32_t block_height = OMP_COMPRESS_TILE_HEIGHT;
	switch (image_info.compress_mode)
	{
	case tiff::CompressionMode::COMPRESSIONMODE_NONE:
		tiffCompression = COMPRESSION_NONE;
		block_height = image_info.height;
		break;
	case tiff::CompressionMode::COMPRESSIONMODE_LZW:
		tiffCompression = COMPRESSION_LZW;
		tiffPredictor = PREDICTOR_HORIZONTAL;
		break;
	case  tiff::CompressionMode::COMPRESSIONMODE_JPEG:
		tiffCompression = COMPRESSION_JPEG;
		block_height = image_info.height;
		break;
	case  tiff::CompressionMode::COMPRESSIONMODE_ZIP:
		tiffCompression = COMPRESSION_DEFLATE;
		break;
	default:
		return ErrorCode::ERR_COMPRESS_TYPE_NOTSUPPORT;
	}

	info.bits_per_sample = image_info.valid_bits;
	info.block_width = image_info.width;
	info.block_height = block_height;
	info.image_width = image_info.width;
	info.image_height = image_info.height;
	info.compression = tiffCompression;
	info.predictor = tiffPredictor;
	info.planarconfig = PLANARCONFIG_CONTIG;
	info.photometric = PHOTOMETRIC_MINISBLACK;
	info.image_byte_count = (image_info.valid_bits + 7) / 8;
	info.samples_per_pixel = image_info.samples_per_pixel;
	if (info.samples_per_pixel == 3 && image_info.image_type == tiff::ImageType::IMAGE_RGB)
	{
		info.photometric = tiffCompression == COMPRESSION_JPEG ? PHOTOMETRIC_YCBCR : PHOTOMETRIC_RGB;
	}

	return ErrorCode::STATUS_OK;
}

int32_t tiff_single::open_tiff(const wchar_t* file_name, tiff::OpenMode mode)
{
	wchar_t full_file_name[_MAX_PATH] = L"";
	wchar_t* ret = _wfullpath(full_file_name, file_name, _MAX_PATH);
	wchar_t tiff_file_dir[_MAX_PATH] = L"";
	wchar_t* name = wcsrchr(full_file_name, L'\\');
	if (name) {
		name++;
		wmemcpy(tiff_file_dir, full_file_name, name - full_file_name - 1);
	}

	if (0 != _waccess(tiff_file_dir, 0))
	{
		if (mode == tiff::OpenMode::CREATE_MODE) {
			int ret = _wmkdir(tiff_file_dir);
			if (ret != 0)
				return ErrorCode::ERR_FILE_PATH_ERROR;
		}
		else
		{
			return ErrorCode::ERR_FILE_PATH_ERROR;
		}
	}
	if (!name) {
		return ErrorCode::ERR_FILE_PATH_ERROR;
	}

	//wchar_t* extension = wcschr(name, L'.') + 1;
	//if (!extension || !(extension != L"tiff" && extension != L"tif"))
	//	return ErrorCode::ERR_FILE_PATH_ERROR;

	uint8_t open_flag = OPENFLAG_READ;
	switch (mode)
	{
	case tiff::OpenMode::CREATE_MODE:
		open_flag = OPENFLAG_CREATE | OPENFLAG_WRITE;
		break;
	case tiff::OpenMode::READ_ONLY_MODE:
		open_flag = OPENFLAG_READ;
		break;
	case tiff::OpenMode::READ_WRITE_MODE:
		open_flag = OPENFLAG_READ | OPENFLAG_WRITE;
		break;
	default:
		return ErrorCode::ERR_OPENMODE;
	}

	int32_t hdl = micro_tiff_Open(full_file_name, open_flag);
	if (hdl < 0) {
		return hdl;
	}
	_openMode = mode;
	_hdl = hdl;

	return ErrorCode::STATUS_OK;
}

int32_t tiff_single::close_tiff()
{
	if (_openMode != tiff::OpenMode::READ_ONLY_MODE)
	{
		int32_t image_count = micro_tiff_GetIFDSize(_hdl);
		for (int32_t i = 0; i < image_count; i++)
		{
			micro_tiff_CloseIFD(_hdl, i);
		}
	}
	micro_tiff_Close(_hdl);
	return ErrorCode::STATUS_OK;
}

int32_t tiff_single::create_image(tiff::SingleImageInfo image_info)
{
	if (_openMode == tiff::OpenMode::READ_ONLY_MODE)
		return ErrorCode::ERR_OPENMODE;

	if (image_info.compress_mode == tiff::CompressionMode::COMPRESSIONMODE_JPEG && image_info.pixel_type != tiff::PixelType::PIXEL_UINT8)
		return ErrorCode::ERR_DATA_TYPE_NOTSUPPORT;
	if (image_info.pixel_type != tiff::PixelType::PIXEL_UINT8 && image_info.pixel_type != tiff::PixelType::PIXEL_UINT16)
		return ErrorCode::ERR_DATA_TYPE_NOTSUPPORT;

	ImageInfo info;
	info_conversion(image_info, info);

	int32_t ifd_no = micro_tiff_CreateIFD(_hdl, info);
	return ifd_no;
}

int32_t tiff_single::save_image_data(uint32_t image_number, void* image_data, uint32_t stride)
{
	if (_openMode == tiff::OpenMode::READ_ONLY_MODE)
		return ErrorCode::ERR_OPENMODE;

	ImageInfo info;
	int32_t status = micro_tiff_GetImageInfo(_hdl, image_number, info);
	if (status != ErrorCode::STATUS_OK) {
		return status;
	}

	//size_t size = strlen(make);
	//micro_tiff_tiffSetTag(_hdl, _ifd_no, TIFFTAG_MAKE, TagDataType::TIFF_ASCII, (uint32_t)size, (void*)make);

	uint32_t block_pixel_width = info.image_width * info.samples_per_pixel;		//RGB(3), GRAY(1)
	uint32_t block_stride = info.image_byte_count * block_pixel_width;
	uint64_t block_size = info.image_height * static_cast<uint64_t>(block_stride);

	if (stride == 0)
		stride = block_stride;

	if (block_stride > stride)
	{
		return ErrorCode::ERR_STRIDE_NOT_CORRECT;
	}
	if (block_size > SIZE_MAX)
	{
		return ErrorCode::ERR_BUFFER_SIZE_ERROR;
	}

	void* buf = image_data;
	std::unique_ptr<void, std::function<void(void*)>> auto_buffer(nullptr, free);
	if (block_stride != stride)
	{
		auto_buffer.reset(malloc(static_cast<size_t>(block_size)));
		buf = auto_buffer.get();
		if (buf == nullptr)
			return ErrorCode::ERR_BUFFER_IS_NULL;

		for (uint32_t i = 0; i < info.image_height; i++)
		{
			memcpy_s((uint8_t*)buf + (i * block_stride), block_stride, (uint8_t*)image_data + (i * stride), stride);
		}
	}

	switch (info.compression)
	{
	case COMPRESSION_NONE:
		status = micro_tiff_SaveBlock(_hdl, image_number, 0, block_size, buf);
		break;
	case COMPRESSION_LZW:
		status = save_with_lzw_horidif(_hdl, image_number, buf, info);
		break;
	case COMPRESSION_JPEG:
		break;
	case  COMPRESSION_DEFLATE:
		status = save_with_zlib(_hdl, image_number, buf, info);
		break;
	default:
		status = ErrorCode::ERR_COMPRESS_TYPE_NOTSUPPORT;
		break;
	}

	return status;
}

int32_t tiff_single::get_image_info(uint32_t image_number, tiff::SingleImageInfo* info)
{
	if (_openMode == tiff::OpenMode::CREATE_MODE)
		return ErrorCode::ERR_OPENMODE;

	ImageInfo image_info;
	int32_t status = micro_tiff_GetImageInfo(_hdl, image_number, image_info);
	if (status != ErrorCode::STATUS_OK)
	{
		return status;
	}

	//only support uint8_t and uint16_t
	if (image_info.image_byte_count == 1)
		info->pixel_type = tiff::PixelType::PIXEL_UINT8;
	else if (image_info.image_byte_count == 2)
		info->pixel_type = tiff::PixelType::PIXEL_UINT16;
	else
		return ErrorCode::ERR_DATA_TYPE_NOTSUPPORT;

	info->valid_bits = image_info.bits_per_sample;
	info->width = image_info.image_width;
	info->height = image_info.image_height;
	info->samples_per_pixel = image_info.samples_per_pixel;
	if (image_info.samples_per_pixel == 1)
	{
		info->image_type = tiff::ImageType::IMAGE_GRAY;
	}
	else if (image_info.samples_per_pixel == 3 && (image_info.photometric == PHOTOMETRIC_RGB || image_info.photometric == PHOTOMETRIC_YCBCR))
	{
		info->image_type = tiff::ImageType::IMAGE_RGB;
	}
	else
	{
		info->image_type = tiff::ImageType::IMAGE_FLIM;
	}

	switch (image_info.compression)
	{
	case COMPRESSION_NONE:
		info->compress_mode = tiff::CompressionMode::COMPRESSIONMODE_NONE;
		break;
	case COMPRESSION_JPEG:
		info->compress_mode = tiff::CompressionMode::COMPRESSIONMODE_JPEG;
		break;
	case COMPRESSION_LZW:
		info->compress_mode = tiff::CompressionMode::COMPRESSIONMODE_LZW;
		break;
	case COMPRESSION_ADOBE_DEFLATE:
	case COMPRESSION_DEFLATE:
		info->compress_mode = tiff::CompressionMode::COMPRESSIONMODE_ZIP;
		break;
	default:
		return ErrorCode::ERR_COMPRESS_TYPE_NOTSUPPORT;
	}

	return ErrorCode::STATUS_OK;
}

int32_t tiff_single::load_image_data(uint32_t image_number, void* image_data, uint32_t stride)
{
	if (_openMode == tiff::OpenMode::CREATE_MODE) {
		return ErrorCode::ERR_OPENMODE;
	}

	bool is_big_endian = false;
	ImageInfo image_info;
	int32_t status = micro_tiff_GetImageInfo(_hdl, image_number, image_info);
	if (status != ErrorCode::STATUS_OK) {
		return status;
	}

	tiff::PixelType pixel_type = image_info.image_byte_count == 1 ? tiff::PixelType::PIXEL_UINT8 : tiff::PixelType::PIXEL_UINT16;

	uint32_t buffer_pixel_width = image_info.image_width * image_info.samples_per_pixel;
	uint32_t buffer_stride = buffer_pixel_width * image_info.image_byte_count;

	uint32_t complete_block_pixel_width = image_info.block_width * image_info.samples_per_pixel;
	uint32_t complete_block_stride = complete_block_pixel_width * image_info.image_byte_count;

	uint32_t complete_block_size = complete_block_stride * image_info.block_height;

	if (stride == 0)
		stride = buffer_stride;

	if (stride < buffer_stride)
		return ErrorCode::ERR_STRIDE_NOT_CORRECT;

	uint32_t dst_buffer_width = stride / image_info.image_byte_count;

	uint32_t rows = (uint32_t)ceil((double)image_info.image_height / image_info.block_height);
	uint32_t columns = (uint32_t)ceil(image_info.image_width / image_info.block_width);

	uint32_t total_count = rows * columns;

	if (image_info.compression == COMPRESSION_NONE)
	{
		unique_ptr<void, function<void(void*)>> auto_raw_block_buffer(malloc(complete_block_size), free);
		void* raw_block_buf = auto_raw_block_buffer.get();
		if (raw_block_buf == nullptr)
			return ErrorCode::ERR_BUFFER_IS_NULL;

		for (int i = 0; i < (int)total_count; i++)
		{
			int column = i % columns;
			int row = i / columns;

			uint32_t block_width = image_info.block_width * (column + 1) > image_info.image_width ? image_info.image_width - image_info.block_width * column : image_info.block_width;
			uint32_t block_stride = block_width * image_info.image_byte_count * image_info.samples_per_pixel;

			uint32_t block_height = image_info.block_height * (row + 1) > image_info.image_height ? image_info.image_height - image_info.block_height * row : image_info.block_height;
			uint32_t block_size = block_height * block_stride;

			uint32_t block_no = (uint32_t)i;

			uint64_t size = complete_block_size;
			status = micro_tiff_LoadBlock(_hdl, image_number, block_no, size, raw_block_buf);
			if (status == ErrorCode::STATUS_OK)
			{
				for (uint32_t h = 0; h < block_height; h++)
				{
					uint8_t* src_ptr = (uint8_t*)raw_block_buf + h * block_width * image_info.samples_per_pixel;
					uint8_t* dst_ptr = (uint8_t*)image_data + (row * image_info.block_height + h) * stride + column * complete_block_pixel_width;
					memcpy_s(dst_ptr, block_width * image_info.samples_per_pixel, src_ptr, block_width * image_info.samples_per_pixel);
				}
			}
			else
				return status;
		}
		return ErrorCode::STATUS_OK;
	}

	//Get max block byte size
	uint64_t max_block_byte_size = 0;
	for (uint32_t i = 0; i < total_count; i++)
	{
		uint64_t size = 0;
		status = micro_tiff_LoadBlock(_hdl, image_number, i, size, nullptr);
		if (status != ErrorCode::STATUS_OK)
			return status;
		if (size == 0)
			return ErrorCode::TIFF_ERR_BLOCK_SIZE_IN_IFD_IS_EMPTY;
		if (size > max_block_byte_size)
			max_block_byte_size = size;
	}

	int32_t threads = (min)(omp_get_num_procs() / 2, (int32_t)total_count);
	threads = (max)(1, threads);

	unique_ptr<void, function<void(void*)>> auto_block_buffer(malloc(complete_block_size * threads), free);
	unique_ptr<void, function<void(void*)>> auto_encode_buffer(malloc((size_t)(max_block_byte_size * threads)), free);

	void* block_buf = auto_block_buffer.get();
	void* encode_buf = auto_encode_buffer.get();

	if (block_buf == nullptr || encode_buf == nullptr)
		return ErrorCode::ERR_BUFFER_IS_NULL;

	if (!omp_in_parallel())
		omp_set_num_threads(threads);

#pragma omp parallel for
	for (int i = 0; i < (int)total_count; i++)
	{
		if (status != ErrorCode::STATUS_OK)
			continue;

		int column = i % columns;
		int row = i / columns;

		uint32_t block_width = image_info.block_width * (column + 1) > image_info.image_width ? image_info.image_width - image_info.block_width * column : image_info.block_width;
		uint32_t block_stride = block_width * image_info.image_byte_count * image_info.samples_per_pixel;

		uint32_t block_height = image_info.block_height * (row + 1) > image_info.image_height ? image_info.image_height - image_info.block_height * row : image_info.block_height;
		uint32_t block_size = block_height * block_stride;

		uint32_t block_no = (uint32_t)i;

		int thread_num = omp_get_thread_num();
		uint8_t* thread_encode_buf = (uint8_t*)encode_buf + max_block_byte_size * thread_num;
		uint8_t* thread_decode_buf = (uint8_t*)block_buf + complete_block_size * thread_num;

		uint64_t encode_size = max_block_byte_size;
		status = micro_tiff_LoadBlock(_hdl, image_number, block_no, encode_size, thread_encode_buf);
		if (status != ErrorCode::STATUS_OK) {
			continue;
		}

		switch (image_info.compression)
		{
		case COMPRESSION_JPEG:
			status = ErrorCode::ERR_COMPRESS_TYPE_NOTSUPPORT;
			//int width, height, samples;
			//status = jpeg_decompress((unsigned char*)thread_decode_buf, (unsigned char*)thread_encode_buf, (unsigned long)encode_size, &width, &height, &samples);
			//size = (uint64_t)width * height * samples;
			break;
		case COMPRESSION_LZW:
			try
			{
				uint64_t decode_size = LZWDecode(thread_encode_buf, encode_size, thread_decode_buf, block_size);
				if (decode_size == block_size)
					status = ErrorCode::STATUS_OK;
				else
					status = ErrorCode::ERR_BUFFER_SIZE_ERROR;
				if (status == ErrorCode::STATUS_OK && image_info.predictor == PREDICTOR_HORIZONTAL)
				{
					int hori_status = horizontal_acc(thread_decode_buf, block_height, block_width, image_info.image_byte_count, image_info.samples_per_pixel, is_big_endian);
					if (hori_status != 0)
					{
						status = ErrorCode::ERR_DECOMPRESS_LZW_FAILED;
						break;
					}
				}
			}
			catch (exception e)
			{
				status = ErrorCode::ERR_DECOMPRESS_LZW_FAILED;
				break;
			}
			break;
		case COMPRESSION_ADOBE_DEFLATE:
		case COMPRESSION_DEFLATE:
		{
			status = ErrorCode::ERR_COMPRESS_TYPE_NOTSUPPORT;
			//size = block_size;
			//int zlib_status = uncompress((unsigned char*)thread_decode_buf, (unsigned long*)&size, (unsigned char*)thread_encode_buf, (unsigned long)encode_size);
			//if (zlib_status != 0)
			//{
			//	status = ErrorCode::ERR_DECOMPRESS_ZLIB_FAILED;
			//	break;
			//}
		}
		break;
		default:
			status = ErrorCode::ERR_COMPRESS_TYPE_NOTSUPPORT;
			break;
		}

		if (status == ErrorCode::STATUS_OK)
		{
			for (uint32_t h = 0; h < block_height; h++)
			{
				uint8_t* src_ptr = (uint8_t*)thread_decode_buf + h * block_width * image_info.samples_per_pixel;
				uint8_t* dst_ptr = (uint8_t*)image_data + (row * image_info.block_height + h) * stride + column * complete_block_pixel_width;
				memcpy_s(dst_ptr, block_width * image_info.samples_per_pixel, src_ptr, block_width * image_info.samples_per_pixel);
			}
		}
	}

	return status;
}

int32_t tiff_single::set_image_tag(uint32_t image_number, uint16_t tag_id, uint16_t tag_type, uint32_t tag_count, void* tag_value)
{
	if (_openMode == tiff::OpenMode::READ_ONLY_MODE)
		return ErrorCode::ERR_OPENMODE;

	return micro_tiff_SetTag(_hdl, image_number, tag_id, tag_type, tag_count, tag_value);
}

int32_t tiff_single::get_image_tag(uint32_t image_number, uint16_t tag_id, uint32_t tag_size, void* tag_value)
{
	if (_openMode == tiff::OpenMode::CREATE_MODE)
		return ErrorCode::ERR_OPENMODE;

	uint16_t tag_type = 0;
	uint32_t tag_count = 0;
	int32_t status = micro_tiff_GetTagInfo(_hdl, image_number, tag_id, tag_type, tag_count);
	if (status != ErrorCode::STATUS_OK)
		return status;

	uint8_t type_size = 0;
	switch (tag_type)
	{
	case TagDataType::TIFF_ASCII:
	case TagDataType::TIFF_BYTE:
	case TagDataType::TIFF_SBYTE:
	case TagDataType::TIFF_UNDEFINED:
		type_size = 1;
		break;
	case TagDataType::TIFF_SHORT:
	case TagDataType::TIFF_SSHORT:
		type_size = 2;
		break;
	case TagDataType::TIFF_LONG:
	case TagDataType::TIFF_SLONG:
	case TagDataType::TIFF_FLOAT:
	case TagDataType::TIFF_IFD:
		type_size = 4;
		break;
	case TagDataType::TIFF_RATIONAL:
	case TagDataType::TIFF_SRATIONAL:
	case TagDataType::TIFF_DOUBLE:
	case TagDataType::TIFF_LONG8:
	case TagDataType::TIFF_SLONG8:
	case TagDataType::TIFF_IFD8:
		type_size = 8;
		break;
	}

	if (tag_size < (uint32_t)type_size * tag_count)
		return ErrorCode::ERR_TAG_CONDITION_NOT_MET;

	return micro_tiff_GetTag(_hdl, image_number, tag_id, tag_value);
}

int32_t tiff_single::get_image_count(uint32_t* image_count)
{
	*image_count = micro_tiff_GetIFDSize(_hdl);
	return ErrorCode::STATUS_OK;
}