class Animal:
    species = 'Mammal'

    @classmethod
    def get_species(cls):
        return cls.species

    @classmethod
    def craete_species(cls, new_Species):
       cls.species = new_Species 

    @staticmethod
    def animal_sound():
        return 'Moo'
    

ani = Animal()

print(ani.species)
Animal.craete_species('Reptiles')

print(ani.get_species())