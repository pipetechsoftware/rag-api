def query(
    search: str,
    agent_id: str,
    limit: int = 3,
    ):# -> Any:# -> Any:# -> Any:
    # ...
    return [
        {
            "id": "1",
            "agent_id": agent_id,
            "page_content": "Os carros hatch são conhecidos pelo seu design compacto, porta-malas integrado à cabine e versatilidade no dia a dia. Modelos como Volkswagen Polo, Chevrolet Onix e Honda Fit oferecem boa economia de combustível, dirigibilidade ágil e espaço interno otimizado. Além disso, são populares por seu custo-benefício, manutenção acessível e facilidade de estacionamento, tornando-se uma escolha ideal para quem busca praticidade sem abrir mão do conforto. 🚗💨",
            "similarity": 0.8,
        },
        {
            "id": "2",
            "agent_id": agent_id,
            "page_content": "Os sedãs são carros conhecidos pelo conforto, estabilidade e espaço extra no porta-malas, sendo ideais para viagens e uso familiar. Modelos como Toyota Corolla, Honda Civic e Chevrolet Cruze combinam elegância com desempenho, oferecendo boa dirigibilidade e tecnologia embarcada. Além disso, costumam ter um rodar mais suave e maior isolamento acústico, garantindo uma experiência de condução mais refinada. Perfeitos para quem busca sofisticação sem abrir mão da praticidade. 🚘✨",
            "similarity": 0.7,
        },
        {
            "id": "3",
            "agent_id": agent_id,
            "page_content": "As picapes são veículos robustos e versáteis, ideais tanto para o trabalho pesado quanto para o lazer. Modelos como Toyota Hilux, Ford Ranger e Chevrolet S10 oferecem alta capacidade de carga, tração 4x4 e desempenho confiável em diferentes terrenos. Além disso, as versões mais modernas trazem tecnologia avançada, conforto e conectividade, tornando-se uma opção perfeita para quem precisa de força sem abrir mão da comodidade. 💪🚛🔥",
            "similarity": 0.6,
        },
    ]
    