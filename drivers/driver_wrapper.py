""" Wrap driver functionality.
    author: Daniel Nichols
    date: October 2023
"""
# std imports
from abc import ABC, abstractmethod


class DriverWrapper(ABC):
    """ Abstract base class for driver wrappers. """

    def __init__(self, parallelism_model):
        self.parallelism_model = parallelism_model


    @abstractmethod
    def write_source(self, content, fpath):
        """ Write the given text to the given file. """
        pass

    @abstractmethod
    def compile(self, *binaries, output_path="a.out"):
        """ Compile the given binaries into a single executable. """
        pass

    @abstractmethod
    def run(self, executable):
        """ Run the given executable. """
        pass

    @abstractmethod
    def test_single_output(self, prompt, output, test_driver_file):
        """ Run a single generated output. """
        pass

    @abstractmethod
    def test_all_outputs_in_prompt(self, prompt):
        """ Run all the generated outputs in the given prompt. """
        pass