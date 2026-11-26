#ifndef KYTY_SHIM_TESTS_TEXTURE_CACHE_TEST_ACCESS_H_
#define KYTY_SHIM_TESTS_TEXTURE_CACHE_TEST_ACCESS_H_

#include "graphics/host_gpu/renderer/cache/textureCache.h"

namespace Libs::Graphics {

struct TextureCacheTestAccess {
	static ImageId Create(TextureCache& cache, const ImageInfo& info) {
		Common::ScopedLock lock {cache.m_lock};
		return cache.InsertImage(info);
	}

	static bool QueueDownload(TextureCache& cache, ImageId image) {
		Common::ScopedLock lock {cache.m_lock};
		return cache.DownloadImageMemory(image);
	}

	static void Copy(TextureCache& cache, ImageId destination, ImageId source) {
		Common::ScopedLock lock {cache.m_lock};
		cache.CopyImage(destination, source);
	}
};

}

#endif
