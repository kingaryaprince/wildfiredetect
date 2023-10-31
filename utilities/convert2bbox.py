from sentinelhub import CRS, BBox


def convert2bbox(longitude, latitude, buffer_size=0.05):
    return BBox(
        bbox=[
            longitude - buffer_size,
            latitude - buffer_size,
            longitude + buffer_size,
            latitude + buffer_size,
        ],
        crs=CRS.WGS84,
    )
