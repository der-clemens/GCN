#include <iostream>
#include "util.h"
#include "rsb.h"
#include "Model.h"

void test(const GraphMatrix& ga) {
    auto m = Matrix(6, 3);
    bli_setm(&BLIS_ZERO, &(m.mat));
    m.set(0,0,1);
    m.set(1,1,1);
    m.set(2,2,1);
    m.set(3,0,1);
    m.set(4,1,1);
    m.set(5,2,1);
    auto result = ga.cross(m);
    for(int i = 0; i < m.rows; i++) {
        for(int j = 0; j < m.cols; j++) {
            std::cout << result.get(i, j) << ", ";
        }
        std::cout << std::endl;
    }
}

int main() {
    rsb_lib_init(nullptr);
    bli_thread_set_num_threads( 4 );
#if 0
    std::string graphFile("../graphs/test.csr");
    std::string featureFile("../graphs/test.features");
    std::string labelFile("../graphs/test.labels");
    std::string trainFile("../graphs/test.train");
#else
    std::string graphFile("../graphs/cora.csrra");
    std::string featureFile("../graphs/cora.features");
    std::string labelFile("../graphs/cora.labels");
    std::string trainFile("../graphs/cora.train");
#endif
    auto GA = readGraph(graphFile);
    auto x = readCSV(featureFile, '\t');
    auto y = readCSV(labelFile, '\t');
    auto train = readCSV(trainFile, '\t');

//    test(GraphMatrix(GA));
//    exit(42);

    const auto filters = new size_t[3] {x.cols,1000, 7 };

    auto model = Model(GA, filters, 2);
    model.fit(x, y, train, 150, 0.05, 0.2);
    return 0;
}
