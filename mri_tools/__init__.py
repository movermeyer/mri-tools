__author__ = 'Robbert Harms'
__date__ = '2015-01-01'
__email__ = 'robbert.harms@maastrichtuniversity.nl'
__license__ = "BSD 3-Clause"
__maintainer__ = "Robbert Harms"

VERSION = '0.1.5'
VERSION_STATUS = ''

_items = VERSION.split('-')                                           
VERSION_NUMBER_PARTS = tuple(int(i) for i in _items[0].split('.'))
if len(_items) > 1:
    VERSION_STATUS = _items[1]
__version__ = VERSION
