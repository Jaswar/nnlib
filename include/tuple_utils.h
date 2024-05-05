/**
 * @file tuple_utils.h
 * @brief
 *
 * @author Jan Warchocki
 * @date 05 May 2024
 *
 */

#ifndef NNLIB_TUPLE_UTILS_H
#define NNLIB_TUPLE_UTILS_H

#include <cstdlib>
#include <utility>
#include <vector>
#include <memory>

// functions from https://stackoverflow.com/questions/72085417/how-to-extract-all-tuple-elements-of-given-types-into-new-tuple
// and https://stackoverflow.com/questions/42494715/c-transform-a-stdtuplea-a-a-to-a-stdvector-or-stddeque

template<typename wanted_type, typename T> struct is_wanted_type;

template<typename wanted_type, typename ...Types>
struct is_wanted_type<wanted_type, std::tuple<Types...>> {

    static constexpr bool wanted=(std::is_same_v<wanted_type, Types>
                                    || ...);
};

// Ok, the ith index in the tuple, here's its std::tuple_element type.
// And wanted_element_t is a tuple of all types we want to extract.
//
// Based on which way the wind blows we'll produce either a std::tuple<>
// or a std::tuple<tuple_element_t>.

template<size_t i, typename tuple_element_t,
         typename wanted_element_t,
         bool wanted=is_wanted_type<tuple_element_t, wanted_element_t>::wanted>
struct extract_type {

    template<typename tuple_type>
    static auto do_extract_type(const tuple_type &t)
    {
        return std::tuple<>{};
    }
};


template<size_t i, typename tuple_element_t, typename wanted_element_t>
struct extract_type<i, tuple_element_t, wanted_element_t, true> {

    template<typename tuple_type>
    static auto do_extract_type(const tuple_type &t)
    {
        return std::tuple<tuple_element_t>{std::get<i>(t)};
    }
};

// And now, a simple fold expression to pull out all wanted types
// and tuple-cat them together.

template<typename wanted_element_t, typename tuple_type, size_t ...i>
auto get_type_t(const tuple_type &t, std::index_sequence<i...>)
{
    return std::tuple_cat( extract_type<i,
                                       typename std::tuple_element<i, tuple_type>::type,
                                       wanted_element_t>::do_extract_type(t)... );
}


template<typename ...wanted_element_t, typename ...types>
auto get_type(const std::tuple<types...> &t)
{
    return get_type_t<std::tuple<wanted_element_t...>>(
            t, std::make_index_sequence<sizeof...(types)>());
}

template <class Tuple,
         class T = std::decay_t<std::tuple_element_t<0, std::decay_t<Tuple>>>>
std::vector<T> to_vector(Tuple&& tuple)
{
    return std::apply([](auto&&... elems){
        return std::vector<T>{std::forward<decltype(elems)>(elems)...};
    }, std::forward<Tuple>(tuple));
}

#endif //NNLIB_TUPLE_UTILS_H
