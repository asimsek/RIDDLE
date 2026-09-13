def uuid():
    import ctypes as ct
    import uuid as identifier

    driver = ct.CDLL("libcuda.so.1")
    driver.cuInit.argtypes = [ct.c_uint]
    driver.cuDeviceGet.argtypes = [ct.POINTER(ct.c_int), ct.c_int]

    def check(code):
        if code:
            raise RuntimeError(f"CUDA device identification failed ({code})")

    check(driver.cuInit(0))
    device = ct.c_int()
    check(driver.cuDeviceGet(ct.byref(device), 0))
    values = []
    for function in (driver.cuDeviceGetUuid, getattr(driver, "cuDeviceGetUuid_v2", driver.cuDeviceGetUuid)):
        function.argtypes = [ct.c_void_p, ct.c_int]
        value = (ct.c_ubyte * 16)()
        check(function(ct.byref(value), device.value))
        values.append(bytes(value))
    return ("MIG-" if values[0] != values[1] else "GPU-") + str(identifier.UUID(bytes=values[1]))


if __name__ == "__main__":
    print(uuid())
