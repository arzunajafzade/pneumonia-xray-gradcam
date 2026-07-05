import config, os
print('TRAIN_DIR =', config.TRAIN_DIR)
print('exists =', os.path.exists(config.TRAIN_DIR))
if os.path.exists(config.DATA_DIR):
    print('DATA_DIR contents:', os.listdir(config.DATA_DIR))
else:
    print('DATA_DIR not found:', config.DATA_DIR)
