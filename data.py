import json
import yaml
import random


class RICFileHandler:
    """
    Handles reading and writing of .ric files.
    """
    @staticmethod
    def load_ric_input(path):
        ''' from RIC file -> data (dict)'''
        try:
            with open(path, 'r+', encoding='UTF-8') as file:
                content = file.readlines()
                unwanted = 'type: !!python/object/apply:uilib.fileIO.fileTypes\n'
                if unwanted in content:
                    content = content[:content.index(unwanted)]
                    file.seek(0)
                    file.writelines(content)
                    file.truncate()
                file.seek(0)
                content = yaml.safe_load(file)
            return content['data']
        except FileNotFoundError:
            raise FileNotFoundError(f"File {path} not found.")

    @staticmethod
    def create_ric_input(data_input, path):
        ''' saves data_input -> to RIC file, adds ending:  "type: !!python/object/apply:uilib.fileIO.fileTypes\n- 3\n\nversion: !!python/tuple\n- 0\n- 6\n- 0"'''
        content = {'data': data_input}
        try:
            with open(path, 'w', encoding='UTF-8') as file:
                yaml.dump(content, file, indent=2)
            with open(path, 'a', encoding='UTF-8') as file:
                file.write("type: !!python/object/apply:uilib.fileIO.fileTypes\n- 3\nversion: !!python/tuple\n- 0\n- 6\n- 0"
                           )
        except Exception as e:
            raise IOError(f"Error writing to {path}: {str(e)}")

    @staticmethod
    def create_ric_result(SEED = None, pathJSON = None, pathRIC = None):
        ''' saves vysledky.json -> to RIC file GEN_RESULT.ric, adds ending:  "type: !!python/object/apply:uilib.fileIO.fileTypes\n- 3\n\nversion: !!python/tuple\n- 0\n- 6\n- 0"'''
        if pathJSON is not None and pathRIC is not None:
            with open(pathJSON, 'r') as file:
                data = json.load(file)
            RICFileHandler.create_ric_input(data['input'], pathRIC)
            return
        if SEED is not None:
            with open(f'result_{SEED}.json', 'r') as file:
                data = json.load(file)
            RICFileHandler.create_ric_input(data['input'], f'GEN_RESULT_{SEED}.ric')
        else:
            with open('vysledky.json', 'r') as file:
                data = json.load(file)
            RICFileHandler.create_ric_input(data['input'], 'GEN_RESULT.ric')


class SimulationLimits:
    """
    Manages the creation and validation of limits for simulation parameters.
    """
    @staticmethod
    def nastav_min_max(param, minf, maxf, factor, input_val, else_val, n_digits=5):
        param['factor'] = factor
        if input_val is None:
            param['max'] = round(else_val * maxf, n_digits)
            param['min'] = round(else_val * minf, n_digits)
        else:
            param['max'] = round(input_val * maxf, n_digits)
            param['min'] = round(input_val * minf, n_digits)
        return param

    def output_scoring(maxf, minf, tot_length, user_out=[]):
        if len(user_out) != 14:
            o_burnTime = None
            o_iniKn = None
            o_peakKn = None
            o_avgPress = None
            o_maxPress = None
            o_impulse = None
            o_isp = None
            o_portThroat = None
            o_peakFlux = None
            o_delThrustCoeff = None
            o_idealThrustCoeff = None
            o_propMass = None
            o_volumeLoad = None
            o_length = None
        else:
            o_burnTime = user_out[0]
            o_iniKn = user_out[1]
            o_peakKn = user_out[2]
            o_avgPress = user_out[3]
            o_maxPress = user_out[4]
            o_impulse = user_out[5]
            o_isp = user_out[6]
            o_portThroat = user_out[7]
            o_peakFlux = user_out[8]
            o_delThrustCoeff = user_out[9]
            o_idealThrustCoeff = user_out[10]
            o_propMass = user_out[11]
            o_volumeLoad = user_out[12]
            o_length = user_out[13]

        limits_output = {}

        limits_output['Burn_Time(s)'] = SimulationLimits.nastav_min_max(
            {}, minf, maxf, 6, o_burnTime, 13)
        limits_output['Initial_Kn'] = SimulationLimits.nastav_min_max(
            {}, minf, maxf, 1, o_iniKn, 400.0)
        limits_output['Peak_Kn'] = SimulationLimits.nastav_min_max(
            {}, minf, maxf, 1, o_peakKn, 850.0)
        limits_output['Avg_Chamber_Pressure(Pa)'] = SimulationLimits.nastav_min_max(
            {}, minf, maxf, 4, o_avgPress, 5e6)
        limits_output['Max_Chamber_Pressure(Pa)'] = SimulationLimits.nastav_min_max(
            {}, minf, 1, 4, o_maxPress, 1e7)

        limits_output['Impulse'] = SimulationLimits.nastav_min_max(
            {}, minf, maxf, 2, o_impulse, 1500)
        limits_output['ISP'] = SimulationLimits.nastav_min_max(
            {}, minf, maxf, 2, o_isp, 130)

        limits_output['Port/throat'] = SimulationLimits.nastav_min_max(
            {}, minf, maxf, 2, o_portThroat, 2.1)
        limits_output['Peak_Mass_Flux'] = SimulationLimits.nastav_min_max(
            {}, minf, maxf, 1, o_peakFlux, 700)
        limits_output['Delivered_thrust_coefficient'] = SimulationLimits.nastav_min_max(
            {}, minf, maxf, 1, o_delThrustCoeff, 1.7)
        limits_output['Ideal_thrust_coefficient'] = SimulationLimits.nastav_min_max(
            {}, minf, maxf, 1, o_idealThrustCoeff, 1.6)

        limits_output['Prop_mass'] = SimulationLimits.nastav_min_max(
            {}, minf, maxf, 1, o_propMass, 1)
        limits_output['Prop_length'] = SimulationLimits.nastav_min_max(
            {}, minf, maxf, 1, o_length, tot_length)
        limits_output['Volume_loading'] = SimulationLimits.nastav_min_max(
            {}, minf, maxf, 1, o_volumeLoad, 88)
        return limits_output

    @staticmethod
    def default_limits(naive_tol=12, num_grains=1, g_diameter=None, g_length=None, g_coreDiameter=None,
                       n_exit=None, n_convAngle=None, n_divAngle=None, n_efficiency=None, n_erosionCoeff=None, n_slagCoeff=None,
                       n_throat=None, n_throatLength=None, p_density=None, user_out=[]):
        minf = (100 - 2 * naive_tol) / 100
        maxf = (100 + 2 * naive_tol) / 100
        data = {'input': {'grains': {}, 'nozzle': {},
                          'propellant': {}}, 'output': {'score': {}}}

        d_avg = g_diameter if g_diameter else 0.033
        g_length = g_length if g_length else d_avg * 7 / num_grains

        grains = data['input']['grains']
        grains['properties'] = {}
        grains['properties']['diameter'] = SimulationLimits.nastav_min_max(
            {}, minf, maxf, 5, g_diameter, 0.055)
        grains['properties']['length'] = SimulationLimits.nastav_min_max(
            {}, minf, maxf, 5, g_length, d_avg * 7 / num_grains)
        grains['properties']['coreDiameter'] = SimulationLimits.nastav_min_max(
            {}, minf, maxf, 5, g_coreDiameter, d_avg*1/3)

        # Example nozzle configuration
        nozzle = data['input']['nozzle']
        nozzle['exit'] = SimulationLimits.nastav_min_max(
            {}, minf, maxf, 1, n_exit, d_avg / 2)
        nozzle['convAngle'] = SimulationLimits.nastav_min_max(
            {}, minf, maxf, 1, n_convAngle, 39.0)
        nozzle['divAngle'] = SimulationLimits.nastav_min_max(
            {}, minf, maxf, 1, n_divAngle, 15.0)

        nozzle['efficiency'] = SimulationLimits.nastav_min_max(
            {}, minf, 1, 0, n_efficiency, 1.0)
        nozzle['erosionCoeff'] = SimulationLimits.nastav_min_max(
            {}, minf, maxf, 0, n_erosionCoeff, 0.0)
        nozzle['slagCoeff'] = SimulationLimits.nastav_min_max(
            {}, minf, maxf, 0, n_slagCoeff, 0.0)

        nozzle['throat'] = SimulationLimits.nastav_min_max(
            {}, minf, maxf, 1, n_throat, d_avg*1/3*1/2*minf)
        nozzle['throatLength'] = SimulationLimits.nastav_min_max(
            {}, 0, maxf, 1, n_throatLength, d_avg*1/3)

        # Example propellant configuration
        propellant = data['input']['propellant']
        propellant['density'] = SimulationLimits.nastav_min_max(
            {}, 1, 1, 0, p_density, 1750)

        # Output scoring
        maxf, minf = 1, 1
        data['output']['score'] = SimulationLimits.output_scoring(
            maxf, minf, g_length*num_grains, user_out=user_out)

        return data

    @staticmethod
    def create_limits_from_ric_data(data_in, naive_tol=12,
                                    o_burnTime=None, o_iniKn=None, o_peakKn=None,
                                    o_avgPress=None, o_maxPress=None, o_impulse=None, o_isp=None,
                                    o_portThroat=None, o_peakFlux=None, o_delThrustCoeff=None, o_idealThrustCoeff=None,
                                    o_propMass=None, o_volumeLoad=None, o_length=None):
        '''Extracts limits from given RIC input data'''

        user_out = [o_burnTime, o_iniKn, o_peakKn, o_avgPress, o_maxPress, o_impulse, o_isp, o_portThroat,
                    o_peakFlux, o_delThrustCoeff, o_idealThrustCoeff, o_propMass, o_volumeLoad, o_length]

        limits = SimulationLimits.default_limits(naive_tol, user_out=user_out)

        for key_type, values in data_in.items():
            if key_type == 'config':
                continue
            if key_type == 'grains':
                for grain in values:
                    for prop, val in grain['properties'].items():
                        if type(val) is str:
                            continue
                        limits['input']['grains']['properties'][prop] = {
                            'min': val * (100-naive_tol)/100,
                            'max': val * (100+naive_tol)/100,
                            'factor': 1
                        }
            else:
                for prop, val in values.items():
                    if type(val) is str or type(val) is list:
                        continue
                    limits['input'][key_type][prop] = {
                        'min': val * (100-naive_tol)/100,
                        'max': val * (100+naive_tol)/100,
                        'factor': 1
                    }

        # limits['output'] = {'score': {}}
        return limits


class DataGenerator:
    """
    Generates input and output data for simulations based on specified limits.
    """
    @staticmethod
    def create_data_from_limits(limits, data_in):
        'Creates random inputdata from limits'
        for key_type, values in limits['input'].items():
            if key_type == 'grains':
                for prop, prop_limits in values['properties'].items():
                    for grain in data_in[key_type]:
                        grain['properties'][prop] = round(
                            random.uniform(
                                prop_limits['min'], prop_limits['max']), 4
                        )
            else:
                for prop, prop_limits in values.items():
                    data_in[key_type][prop] = round(
                        random.uniform(
                            prop_limits['min'], prop_limits['max']), 4
                    )
        return data_in

    @staticmethod
    def create_data(data_file, ric_path, out_path = None, limits='Default', naive_tol =12) -> dict:
        """
        Creates a data dictionary from input/output files and limits.

        Args:
            data_file (str or None): Path to save the generated data file.
            ric_path (str): Path to the input .ric file.
            out_path (str, optional): Path to a JSON file containing output scores.
            limits (str or dict, optional): Either 'Default', 'RIClike' or a custom limits dictionary.
            naive_tol (int, optional): Tolerance used for limits creation.

        Returns:
            dict: A structured dictionary containing info, input, and optional output.
        """
        data = {
            'info': {'id': 1, 'total_score': None},
            'input': {},
            'output': {}
        }

        # Load input data from RIC
        ric_data = RICFileHandler.load_ric_input(ric_path)
        data['input'] = ric_data

        # Apply limits
        if limits == 'Default':
            data['info']['limits'] = SimulationLimits.default_limits(naive_tol)
        elif limits == 'RIClike':
            data['info']['limits'] = SimulationLimits.create_limits_from_ric_data(ric_data, naive_tol)
        else:
            data['info']['limits'] = limits

        # Load optional output score
        if out_path is not None:
            with open(out_path, 'r') as file:
                data['output']['score'] = json.load(file)

        # Save final data to file if specified
        if data_file is not None:
            with open(data_file, 'w') as file:
                json.dump(data, file, indent=4)

        return data


# Example Usage
if __name__ == "__main__":
    # Set paths and parameters
    ric_path = "new.ric"
    out_path = "output_vals.txt"
    data_file = "data.json"

    # Load and generate data
    try:
        datain = RICFileHandler.load_ric_input("new.ric")
        limits = SimulationLimits.create_limits_from_ric_data(data_in=datain)
        # data = DataGenerator.create_data(data_file, ric_path, out_path, limits)
        data = DataGenerator.create_data(
            None, "new.ric", None, limits)
        # limits = SimulationLimits
        print("Data generated successfully.")
        print(limits)
    except Exception as e:
        print(f"Error: {str(e)}")
