import bi
import joblib

def main():
    graph = joblib.load('./data/Twitter/twitter.pkl.cmp')
    bi.make_graph_data_distribution(graph, 'Twitter/twitter.csv')
    bi.convert_csv2image('./data/Twitter/twitter.csv')


if __name__ == '__main__':
    main()
