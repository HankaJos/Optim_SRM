from PyQt6.QtCore import QObject, pyqtSignal

from motorlib.properties import PropertyCollection, FloatProperty, IntProperty, EnumProperty
from motorlib.units import unitLabels, getAllConversions
from motorlib.motor import MotorConfig

from .fileIO import loadFile, saveFile, getConfigPath, fileTypes
from .defaults import DEFAULT_PREFERENCES
from .widgets import preferencesMenu
from .logger import logger

class Preferences():
    '''Class Preferences v sobe zahrnuje dictionaries 'general' a 'units',
    obecne se nacitaji veci (from .defaults import DEFAULT_PREFERENCES), ktere
    nejsou metricke. Na druhou stranu jakmile ulozime preferences.yaml, coz
    se u me dava do slozky (home/hana/.local/share/openMotor), mame to nastavene'''
    def __init__(self, propDict=None):
        self.general = MotorConfig()
        self.units = PropertyCollection()
        for unit in unitLabels:
            self.units.props[unit] = EnumProperty(unitLabels[unit], getAllConversions(unit))

        if propDict is not None:
            self.applyDict(propDict)

    def getDict(self):
        prefDict = {}
        prefDict['general'] = self.general.getProperties()
        prefDict['units'] = self.units.getProperties()
        return prefDict

    def applyDict(self, dictionary):
        self.general.setProperties(dictionary['general'])
        self.units.setProperties(dictionary['units'])

    def getUnit(self, fromUnit):
        if fromUnit in self.units.props:
            return self.units.getProperty(fromUnit)
        return fromUnit


class PreferencesManager(QObject):

    preferencesChanged = pyqtSignal(object)

    def __init__(self, makeMenu=True):
        super().__init__()
        self.preferences = Preferences(DEFAULT_PREFERENCES)
        if makeMenu:
            self.menu = preferencesMenu.PreferencesMenu()
            self.menu.preferencesApplied.connect(self.newPreferences)
        self.loadPreferences()

    def newPreferences(self, prefDict):
        logger.log('Updating preferences')
        self.preferences.applyDict(prefDict)
        self.savePreferences()
        self.publishPreferences()


    def loadPreferences(self):
        ''' soubor preferences.yaml vypada nasledovne:
        data:
        general:
            ambPressure: 101325.0
            burnoutThrustThres: 0.1
            burnoutWebThres: 2.5400050800101604e-05
            flowSeparationWarnPercent: 0.05
            mapDim: 750
            maxMassFlux: 1406.469761
            maxPressure: 10342500.0
            minPortThroat: 2.0
            sepPressureRatio: 0.4
            timestep: 0.001
        units:
            (m*Pa)/s: (m*MPa)/s
            N: N
            Ns: Ns
            Pa: MPa
            kg: kg
            kg/(m^2*s): kg/(m^2*s)
            kg/m^3: kg/m^3
            kg/s: kg/s
            m: mm
            m/(s*Pa): m/(s*MPa)
            m/(s*Pa^n): m/(s*Pa^n)
            m/s: m/s
            m^3: cm^3
        type: !!python/object/apply:uilib.fileIO.fileTypes
        - 1
        version: !!python/tuple
        - 0
        - 6
        - 0
        '''
        try:
            prefDict = loadFile(getConfigPath() + 'preferences.yaml', fileTypes.PREFERENCES)
            self.preferences.applyDict(prefDict)
            self.publishPreferences()
        except FileNotFoundError:
            logger.warn('Unable to load preferences, creating new file')
            self.savePreferences()

    def savePreferences(self):
        try:
            logger.log('Saving preferences to "{}"'.format(getConfigPath() + 'preferences.yaml'))
            saveFile(getConfigPath() + 'preferences.yaml', self.preferences.getDict(), fileTypes.PREFERENCES)
        except:
            logger.warn('Unable to save preferences')

    def showMenu(self):
        logger.log('Showing preferences menu')
        self.menu.load(self.preferences)
        self.menu.show()

    def publishPreferences(self):
        self.preferencesChanged.emit(self.preferences)
