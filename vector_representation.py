from manim import *
import numpy as np
import numpy.typing as npt


np.random.seed(0)
class VectorSpace(Scene):
    def incialize_named_vector(self, vector: npt.NDArray[np.float64], name:str, font_size:float = 35, color: ManimColor = WHITE) -> (Vector, Text):
        vector_obj = Vector(vector, color=color)
        vector_name_obj = Text(name, font_size=font_size, color=color)
        pos = vector_obj.get_end()
        if vector[0] >  0: # if pointing right
            pos -= vector_name_obj.get_left()
        else: # if pointing left
            pos -= vector_name_obj.get_right()
        vector_name_obj.shift(pos)
        return (vector_obj, vector_name_obj)

    def construct(self):
        plane = NumberPlane(x_range=(-28, 28, 1), y_range=(-16, 16, 1))
        self.add(plane)
        
        swimming_array = np.array([0.2, 2])
        running_array = np.array([4.6,2.8])
        animal_array = np.array([4.56,-2.7])
        finance_array = np.array([-4, .5])
        
        running_vector, running_text = self.incialize_named_vector(running_array, "Running", color = YELLOW)
        self.add(running_vector, running_text) 
        
        swimming_vector, swimming_text = self.incialize_named_vector(swimming_array, "Swimming", color = PURPLE_B)
        self.add(swimming_vector, swimming_text)
        
        animal_vector, animal_text = self.incialize_named_vector(animal_array, "Animal", color = RED)
        self.add(animal_vector, animal_text)

        finance_vector, finance_text = self.incialize_named_vector(finance_array, "Finance", color = ORANGE)
        self.add(finance_vector, finance_text)
        
        duathlon_array = running_array-swimming_array + (np.random.random(2)*2 - 1)
        duathlon_array = duathlon_array[:2]
        duathlon_vector, duathlon_text = self.incialize_named_vector(duathlon_array, "Duathlon", color = GREEN)
        self.add(duathlon_vector, duathlon_text)


        swimming_vector.generate_target()
        swimming_vector.target.shift(duathlon_vector.get_end())
        swimming_text.generate_target()
        swimming_text.target.shift(duathlon_vector.get_end())

        self.play(MoveToTarget(swimming_vector), MoveToTarget(swimming_text), run_time=5)
        self.wait(2)