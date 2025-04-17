from configs import cfg

class DatasetArgs(object):
    dataset_attrs = {}

    subjects = ['313', '315', '377', '386', '387', '390', '392', '393', '394']
    
    if cfg.category == 'human_nerf' and cfg.task == 'zju_mocap_blur':
        for sub in subjects:
            if cfg.eval:
                # evaluation
                dataset_attrs.update({
                        f"zju_{sub}_train": {
                            "dataset_path": f"dataset/zju_mocap_blur/processed/{sub}",
                            "keyfilter": cfg.train_keyfilter,
                            "ray_shoot_mode": cfg.train.ray_shoot_mode,
                        },
                        f"zju_{sub}_test": {
                            "dataset_path": f"dataset/zju_mocap_blur/processed/{sub}_eval", 
                            "keyfilter": cfg.test_keyfilter,
                            "ray_shoot_mode": 'image',
                            "src_type": 'zju_mocap',
                            "gama": 0,
                        },
                    })
            else:
                dataset_attrs.update({
                    f"zju_{sub}_train": {
                        "dataset_path": f"dataset/zju_mocap_blur/{sub}",
                        "keyfilter": cfg.train_keyfilter,
                        "ray_shoot_mode": cfg.train.ray_shoot_mode,
                    },
                    f"zju_{sub}_test": {
                        "dataset_path": f"dataset/zju_mocap_blur/{sub}", 
                        "keyfilter": cfg.test_keyfilter,
                        "ray_shoot_mode": 'image',
                        "src_type": 'zju_mocap'
                    },
                })
                # movement freeview
                # dataset_attrs.update({
                #     f"zju_{sub}_train": {
                #         "dataset_path": f"dataset/zju_mocap/{sub}",
                #         "keyfilter": cfg.train_keyfilter,
                #         "ray_shoot_mode": cfg.train.ray_shoot_mode,
                #     },
                #     f"zju_{sub}_test": {
                #         "dataset_path": f"dataset/zju_mocap/{sub}", 
                #         "keyfilter": cfg.test_keyfilter,
                #         "ray_shoot_mode": 'image',
                #         "src_type": 'zju_mocap'
                #     },
                # })

    @staticmethod
    def get(name):
        attrs = DatasetArgs.dataset_attrs[name]
        return attrs.copy()
