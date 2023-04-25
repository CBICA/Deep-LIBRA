The toolkit has been tested on Ubuntu 20.04, Ubuntu 18.04, Centos 7, cluster based on Centos 7, and MacOX. It is expected to work on Windows too. Python 3 is needed for running the toolkit. Please follow the following instructions to install and use software:

1. Install Conda from the following link: https://docs.conda.io/projects/conda/en/latest/user-guide/install/
2. Set up a virtual environment:

       conda create --name DeepLIBRA python=3.6
3. Activate the environment for the first time and anytime you need to work with the toolkit:

       conda activate DeepLIBRA
4. Install the required packages:

       pip install -r requirements.txt
5. The toolkit can be run either from the command line or using a graphical used interface:
    * Run toolkit using command line (there more options for running the toolkit, you can see them using `-h`):

          python3 Path_to_code/execute_libra_code.py -i ${Path_to_Images} -o ${Path_to_Output} -m ${Path_to_Networks}

      Example:

          python3 ~/Documents/MyPapers/A_Breast_Deep_LIBRA/ShortVersion/Code/execute_libra_code.py -i ~/Desktop/image -o ~/Desktop/image/output -m ~/Desktop/Nets
    * Run toolkit using graphical user interface:

          python3 ${Path_to_code}/run_GUI.py
          
The pretrained networks can be found from this link: https://upenn.box.com/s/y08cpr0soxxu05x5godw7h5czpn7oak8
