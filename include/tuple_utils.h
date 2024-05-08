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
#include <memory>
#include <utility>
#include <vector>

// functions from https://stackoverflow.com/questions/72085417/how-to-extract-all-tuple-elements-of-given-types-into-new-tuple
// and https://stackoverflow.com/questions/42494715/c-transform-a-stdtuplea-a-a-to-a-stdvector-or-stddeque

template<typename wantedType, typename T>
struct IsWantedType;

template<typename wantedType, typename... Types>
struct IsWantedType<wantedType, std::tuple<Types...>> {

    static constexpr bool wanted = (std::is_same_v<wantedType, Types> || ...);
};

// Ok, the ith index in the tuple, here's its std::tuple_element type.
// And wanted_element_t is a tuple of all types we want to extract.
//
// Based on which way the wind blows we'll produce either a std::tuple<>
// or a std::tuple<tuple_element_t>.

template<size_t i, typename tuple_element_t, typename wanted_element_t,
         bool wanted = IsWantedType<tuple_element_t, wanted_element_t>::wanted>
struct ExtractType {

    template<typename tuple_type>
    static auto doExtractType(const tuple_type& t) {
        return std::tuple<>{};
    }
};


template<size_t i, typename tuple_element_t, typename wanted_element_t>
struct ExtractType<i, tuple_element_t, wanted_element_t, true> {

    template<typename tuple_type>
    static auto doExtractType(const tuple_type& t) {
        return std::tuple<tuple_element_t>{std::get<i>(t)};
    }
};

// And now, a simple fold expression to pull out all wanted types
// and tuple-cat them together.

template<typename wanted_element_t, typename tuple_type, size_t... i>
auto getTypeT(const tuple_type& t, std::index_sequence<i...>) {
    return std::tuple_cat(
            ExtractType<i, typename std::tuple_element<i, tuple_type>::type, wanted_element_t>::doExtractType(t)...);
}


template<typename... wanted_element_t, typename... types>
auto getType(const std::tuple<types...>& t) {
    return getTypeT<std::tuple<wanted_element_t...>>(t, std::make_index_sequence<sizeof...(types)>());
}

template<class Tuple, class T = std::decay_t<std::tuple_element_t<0, std::decay_t<Tuple>>>>
std::vector<T> toVector(Tuple&& tuple) {
    return std::apply([](auto&&... elems) { return std::vector<T>{std::forward<decltype(elems)>(elems)...}; },
                      std::forward<Tuple>(tuple));
}

#endif //NNLIB_TUPLE_UTILS_H
