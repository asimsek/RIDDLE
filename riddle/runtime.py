from concurrent.futures import ThreadPoolExecutor


def ordered_map(function, items, workers=1):
    items = list(items)
    if workers <= 1 or len(items) < 2:
        return [function(item) for item in items]
    with ThreadPoolExecutor(max_workers=min(workers, len(items))) as pool:
        return list(pool.map(function, items))


def fingerprints(paths, workers=1, progress=None):
    from riddle.storage import file_digest

    paths = list(paths)
    result = {}
    with ThreadPoolExecutor(max_workers=max(1, min(workers, len(paths)))) as pool:
        for index, (path, value) in enumerate(zip(paths, pool.map(file_digest, paths)), 1):
            result[path] = value
            if progress is not None:
                progress.update(index)
    return result
