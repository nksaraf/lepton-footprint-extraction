from shapely.geometry import MultiPolygon, Polygon, mapping
import shapely.affinity as aff
import fiona
from fiona.crs import from_epsg
import rasterio

from model.connections import Transformer

def get_transform(jpg_path):
    with rasterio.open(jpg_path) as jpg:
        scale = (jpg.transform.a, jpg.transform.e)
        translate = jpg.transform * (0, 0)
    return { 'scale': scale, 'translate': translate }

def make_transform(transform):
    scale = (transform.a, transform.e)
    translate = transform * (0, 0)
    return { 'scale': scale, 'translate': translate }

class ShapefileCreator(Transformer):
    __out__ = ('status', )

    def __init__(self, name, crs=None):
        super(ShapefileCreator, self).__init__(name)
        self.schema = {
            'geometry': 'Polygon',
            'properties': {
                'ID': 'int'
            }
        }
        self.crs = crs

    def __transform__(self, filename, polygons, transform={}):
        with fiona.open(filename, 'w', crs=self.crs, driver='ESRI Shapefile', schema=self.schema) as shp:
            if type(polygons) == list:
                pass
            else:
                self.log('Geom type: ' + polygons.geom_type)
                for i, geom in enumerate(polygons.geoms):
                    if 'scale' in transform:
                        geom = aff.scale(geom, xfact=transform['scale'][0], yfact=transform['scale'][1], origin=(0, 0, 0))
                    if 'translate' in transform:
                        geom = aff.translate(geom, xoff=transform['translate'][0], yoff=transform['translate'][1])
                    shp.write({
                        'geometry': mapping(geom),
                        'properties': {
                            'ID': i
                        }
                    })
        return { 'status': True }
