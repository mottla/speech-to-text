import bisect
import os
from audioop import error

import torch
import webcolors
import random

COLORS = list(webcolors.names())


class SortedList:
    def __init__(self,file_path = "learned_speakers.pth"):
        self.elements = []  # List to hold (value, tensor) tuples
        # Get a list of named colors from matplotlib
        # Get a list of named colors from webcolors
        self.file_path = file_path

    def add(self, key ,embedding, gptEncoding,  name = None):
        # Insert the tensor in sorted order based on value
        #bisect.insort(self.elements, (value, tensor))  #ascending
        if name is None:
            # Pick a random color and remove it from the list
            name = random.choice(COLORS)
            COLORS.remove(name)

        bisect.insort(self.elements, (key, embedding, gptEncoding, name), key=lambda x: -x[0])  #decending
        return name

    def update_by_index(self,idx,  embedding, gptEncoding, value):
        # Find the tensor and remove it
        name = self.elements[idx][3]
        del self.elements[idx]
        # Add the tensor back with the new value
        self.add(value,embedding, gptEncoding, name)

    def update_by_name(self,name ,  embedding = None, gptEncoding = None, key = None):
        # Find the tensor and remove it
        for i,(key1, embedding1, gptEncoding1,name1) in enumerate(self.elements):
            if name1 == name:
                self.elements[i] = (key1 if key is None else key, embedding1 if embedding is None else embedding , gptEncoding1 if gptEncoding is None else gptEncoding , name)
                break

    def delete_by_name(self,name ):
        # Find the tensor and remove it
        for i,(_, _, _,name1) in enumerate(self.elements):
            if name1 == name:
                del self.elements[i]
                COLORS.append(name1)

    def delete(self,index):
        del self.elements[index]

    def get(self,index):
        return self.elements[index]

    def getByName(self,name):
        for i,(value1, embedding1, gptEncoding1,name1) in enumerate(self.elements):
            if name1 == name:
                return self.elements[i]

    def len(self):
        return len(self.elements)

    def save(self):
        # Save the sorted list to a file
        # Get the directory name from the file path
        dir_name = os.path.dirname(self.file_path)

        # Save the sorted list to a file
        if dir_name and not os.path.exists(dir_name):
            os.makedirs(dir_name, exist_ok=True)
        torch.save(self.elements, self.file_path)


    def load(self):
        if os.path.exists(self.file_path):
            # Load the sorted list from a file
            self.elements = torch.load(self.file_path)
            for (_,_,_,name) in self.elements:
                if name not in COLORS:
                    print(name)
                else:
                    COLORS.remove(name)
        else:
            print(f"could not find learned speaker voices. Creating new file {self.file_path}")
        return self

    def __iter__(self):
        return iter(self.elements)

    def __reversed__(self):
        return reversed(self.elements)

# Example usage
if __name__ == "__main__":
    sorted_list = SortedList()

    # Adding tensors to the sorted list
    sorted_list.add(torch.tensor([1, 2, 3]),torch.tensor([1]), 5)
    sorted_list.add(torch.tensor([4, 5, 6]),torch.tensor([2]), 3)
    sorted_list.add(torch.tensor([7, 8, 9]),torch.tensor([3]), 8)

    print("Sorted elements:")
    for i, (value, tensor,gpt,name) in enumerate(sorted_list):
        print(f"Tensor {name} at rank {i}: {tensor.numpy()},{gpt.numpy()}, Value: {value}")

    print("Sorted elements:")
    for i, (value, tensor,gpt,name) in enumerate(sorted_list.elements):
        print(f"Tensor {name} at rank {i}: {tensor.numpy()},{gpt.numpy()}, Value: {value}")
    # Update the value of a tensor
    ids= 0
    sorted_list.update_by_index( ids , torch.tensor([9, 9, 9]),torch.tensor([4]), 2)

    print(f"\nAfter updating tensor index {ids}, {sorted_list.get(ids)}")
    for i, (value, tensor,gpt,name) in enumerate(sorted_list):
        print(f"Tensor {name} at rank {i}: {tensor.numpy()}, {gpt.numpy()}, Value: {value}")
    for i, (value, tensor,gpt,name) in enumerate(sorted_list.elements):
        print(f"Tensor {name} at rank {i}: {tensor.numpy()}, {gpt.numpy()}, Value: {value}")
    print(f"Reverse")
    for i, (value, tensor,gpt,name) in enumerate(reversed(sorted_list)):
        print(f"Tensor {name} at rank {i}: {tensor.numpy()}, {gpt.numpy()}, Value: {value}")
    # Save the sorted list to a file
    sorted_list.save()

    # Create a new SortedList and load the saved data
    new_sorted_list = SortedList("test.pth")
    new_sorted_list.load()

    print("\nLoaded elements from file:")
    for i, (value, tensor, gpt, name) in enumerate(sorted_list):
        print(f"Tensor {name} at rank {i}: {tensor.numpy()}, {gpt.numpy()}, Value: {value}")

