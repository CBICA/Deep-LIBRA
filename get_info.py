import os, json



def read_json(obj):
    Path, File = os.path.split(obj.model_path)
    if File.find("_model.h5")>-1:
        File = File[:File.find("_model.h5")]
    else:
        File = File[:File.find("_Model.h5")]
    json_path = os.path.join(Path, File+"_data.json")
    obj.json = json.loads(open(json_path).read())

    return(obj)



def get_info_from_network(obj, model_path, Keys_input=[], Keys_output=[]):
    obj = read_json(obj)

    Path, _ = os.path.split(model_path)

    parameters = dict()
    Parameters_file = open(os.path.join(Path, "parameters.txt"), "r")
    data_in_param_file = Parameters_file.readlines()

    for item in data_in_param_file:
        if item.find('\n')>-1:
            item = item[:item.find('\n')]
        key, value = item.split(':', 1)
        parameters[key]=value


    for Key_i, Key_o in zip(Keys_input, Keys_output):
        try:
            value = parameters[Key_i]
            try:
                setattr(obj, Key_o, int( value ))
            except:
                setattr(obj, Key_o, value)
        except:
            Key_o = "NA"

    return(obj)
