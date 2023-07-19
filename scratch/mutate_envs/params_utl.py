string_property_units ={
    'bottleneck_bandwidth': 'Mbps',
    'bottleneck_delay': 'ms',
    'access_bandwidth': 'Mbps',
    'access_delay': 'ms',
}

float_list = ['bottleneck_delay']

def to_string(EA):
    enviroment = EA[0]
    agent = EA[1]

    for env in list(enviroment.keys()):
        if env in list(string_property_units.keys()):
            enviroment[env] = str(enviroment[env]) + string_property_units[env]
    
    return [enviroment, agent]

def to_number(EA):
    enviroment = EA[0]
    agent = EA[1]

    for env in list(enviroment.keys()):
        if env in list(string_property_units.keys()):
            enviroment[env] = float(enviroment[env].replace(string_property_units[env], ''))
        if enviroment[env] not in float_list:
            enviroment[env] = int(enviroment[env])

    return [enviroment, agent]
