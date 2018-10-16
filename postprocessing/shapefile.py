from shapely.geometry import MultiPolygon, Polygon, mapping
import shapely.affinity as aff
import fiona
from fiona.crs import from_epsg
import rasterio

from model.connections import Transformer

def get_transform(jpg_path):
	"""Extracts transformation parameters from image jpg to be used
	to transform polygons when creating shapefiles from the predicted polygons
	
	Args:
		jpg_path: path to input image (should be geocoded)

	Returns:
		A dictionary with scaling and translating parameters
	"""
    with rasterio.open(jpg_path) as jpg:
        scale = (jpg.transform.a, jpg.transform.e)
        translate = jpg.transform * (0, 0)
    return { 'scale': scale, 'translate': translate }

def make_transform(transform):
	"""Extracts essential transformation parameters from the given
	transformation matrix of to be used to transform polygons when creating 
	shapefiles from the predicted polygons
	
	Args:
		transform: a Rasterio transformation matrix

	Returns:
		A dictionary with scaling and translating parameters
	"""
    scale = (transform.a, transform.e)
    translate = transform * (0, 0)
    return { 'scale': scale, 'translate': translate }

class ShapefileCreator(Transformer):
	"""A transformer that creates a shapefile with the given polygons and saves is to file

	Input:
		filename: path to save shapefile at
		polygons: all polygons to be saved as a MultiPolygon
		transform: transformation matrix got from the ``make_transform`` or ``get_transform`` functions

	Output:
		status: boolean indicating success

	Args:
		crs: Projection system used for shapefiles (should be consistent will other inputs, ideally
				be same as the one used for the input images)
	"""
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
                return { 'status': False }
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
