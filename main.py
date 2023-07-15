from EvolutioaryAlgo import EvolutioaryAlgo

def main():
    ea = EvolutioaryAlgo(.5,5,5,.5)
    bestChromosomsOfAll = ea.run()
    ea.show_result(bestChromosomsOfAll)
    
if __name__ == '__main__':
    main()


