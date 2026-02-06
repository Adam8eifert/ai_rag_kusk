from rag import RAGEngine


def main():
    engine = RAGEngine()
    engine.build_index(data_dir='data', index_dir='index')


if __name__ == '__main__':
    main()
