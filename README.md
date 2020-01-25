# PetNotify

PetNotify.py: main functionality of loading the TensorFlow model and PiCamera image acquisition. Uses notify.run for sending a push notification to any browser. Credit for the groundwork to Evan Juras (https://github.com/EdjeElectronics/TensorFlow-Object-Detection-on-the-Raspberry-Pi).

coco-object-categories.py: pulls the mapping of category numbers to the actual object descriptor from instances_val2017.json and writes it to data.txt.


Note: at this point, this program does not work with numpy versions newer than 1.15. Also, the newest TensorFlow version in the Debian repository is not compatible with Raspberry Pi yet. For a working TensorFlow installation, go here: https://www.tensorflow.org/install/source_rpi
