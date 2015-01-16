#pragma once
template <class T>
struct dictionary {
  std::unordered_map<T, size_t> dict;
  std::vector<T> idict;
  size_t id(T s, bool add = true) { 
    if (!dict.count(s)) {
      if (!add) {
        std::cerr << "dictionary does not have entry " << s << std::endl;
        return -1;
      }
      dict[s] = idict.size();
      idict.push_back(s);
    }
    return dict[s];
  }
  T value(size_t id) {
    return idict[id];
  }
  size_t size() const { 
    return idict.size(); 
  }
};
