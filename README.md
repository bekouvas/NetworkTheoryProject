This is a project that I implemented in Python within the framework of Network Theory class that I attented.
The project concerns the design and implementation of an algorithm that uses the distance quality function for finding communities in networks. As data I used a network that contains 1005 nodes and 25571 edges and is directed! (in file data_sheets you will find the data).

IMPORTANT:
To run the code make sure you run this commants
-pip install networkx
-pip install python-louvain
-pip install numpy

Also update the file paths in main to match the ones in your directory:
detector = CommunityDetector.from_files(
    'path/to/email-Eu-core.txt',
    'path/to/email-Eu-core-department-labels.txt'
)

If using an IDE like PyCharm or VS Code, make sure to set up a Python interpreter with the installed packages
