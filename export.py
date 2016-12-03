import os
import numpy as np

from ddpg_torcs import DDPGTorcs


def convert_h5f_dlj4(actor_model, h5f_filepath, out_filepath):
    actor_filepath = h5f_filepath
    actor_model.load_weights(actor_filepath)
    r = []
    for w in actor_model.get_weights():
        r += np.transpose(w).flatten().tolist()
    np.savetxt(out_filepath, np.array(r))

def convert_all( h5f_folder, dlj4_folder):
    actor_model = DDPGTorcs.get_actor((29,), (2,))
    for file_name in os.listdir(h5f_folder):
        if '.h5f' in file_name:
            out_file=file_name.replace('h5f', 'ffn')
            convert_h5f_dlj4(actor_model, h5f_folder+'/'+file_name,dlj4_folder+'/'+out_file)

convert_all('tested', '/home/marco/Desktop')