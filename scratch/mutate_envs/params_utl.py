string_property_units ={
    'bottleneck_bandwidth': 'Mbps',
    'bottleneck_delay': 'ms',
    'access_bandwidth': 'Mbps',
    'access_delay': 'ms',
}

float_list = ['error_rate', 'bottleneck_bandwidth']

def to_string(enviroment):
    for env in list(enviroment.keys()):
        if env in list(string_property_units.keys()):
            enviroment[env] = str(enviroment[env]) + string_property_units[env]
    
    return enviroment

def to_number(enviroment):
    for env in list(enviroment.keys()):
        if env in list(string_property_units.keys()):
            enviroment[env] = float(enviroment[env].replace(string_property_units[env], ''))
        if env not in float_list:
            enviroment[env] = int(enviroment[env])

    return enviroment
