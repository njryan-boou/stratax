#include <cassert>
#include <sstream>
#include <stdexcept>

#include <stratax.hpp>

using namespace stratax::core;

void test_default_constructor()
{
    Shape shape;

    assert(shape.rank() == 0);
    assert(shape.begin() == shape.end());
}

void test_initializer_list_constructor()
{
    Shape shape{3, 224, 224};

    assert(shape.rank() == 3);
    assert(shape(0) == 3);
    assert(shape(1) == 224);
    assert(shape(2) == 224);
}

void test_index_operater()
{
    Shape shape{3, 224, 224};

    assert(shape(0) == 3);
    assert(shape(1) == 224);
    assert(shape(2) == 224);

    bool threw = false;

    try {
        shape(3);
    }
    catch (const Exceptions::IndexError&) {
        threw = true;
    }

    assert(threw);
}

void test_equality()
{
    Shape a{3, 224, 224};
    Shape b{3, 224, 224};
    Shape c{224, 224};

    assert(a == b);
    assert(!(a != b));

    assert(a != c);
    assert(!(a == c));
}

void test_iteration()
{
    Shape shape{3, 224, 224};

    std::size_t expected[] = {3, 224, 224};
    std::size_t i = 0;

    for (std::size_t dim : shape) {
        assert(dim == expected[i]);
        ++i;
    }

    assert(i == 3);
}

void test_output()
{
    Shape shape{3, 224, 224};

    std::ostringstream out;
    out << shape;

    assert(out.str() == "(3, 224, 224)");
}

int main()
{
    test_default_constructor();
    test_initializer_list_constructor();
    test_index_operater();
    test_equality();
    test_iteration();
    test_output();

    return 0;
}
