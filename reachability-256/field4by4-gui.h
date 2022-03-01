#include <ostream>
#include <string>
#include "field4by4-board-definition.h"


struct board;


#define _BLACK_WHITE_COLOR_SCHEME

#ifdef _BLACK_WHITE_COLOR_SCHEME
std::string background_symbol_for_tile(int tile) {
  switch (tile) {
    case 0:
    case 2:
    case 4:
    case 8:
      return " ";
    case 16:
    case 32:
      return ".";
    case 64:
      return "-";
    case 128:
      return "+";
    default:
      return "!";
  }
}
#endif

#ifdef _COLORFUL_COLOR_SCHEME
std::string background_symbol_for_tile(int tile) {
  return " ";
}
#endif


std::string string_for_tile(int tile) {
  switch (tile) {
    case 0:
      return background_symbol_for_tile(tile)
          + background_symbol_for_tile(tile)
          + background_symbol_for_tile(tile)
          + background_symbol_for_tile(tile);
    case 2:
    case 4:
    case 8:
      return background_symbol_for_tile(tile)
          + (tile == 2 ? "2" : tile == 4 ? "4" : tile == 8 ? "8" : "0")
          + background_symbol_for_tile(tile)
          + background_symbol_for_tile(tile);
    case 16:
    case 32:
    case 64:
      return background_symbol_for_tile(tile)
          + (tile == 16 ? "16" : tile == 32 ? "32" : "64")
          + background_symbol_for_tile(tile);
    default:
      return (tile == 128 ? "128" : tile == 256 ? " 256" : tile == 512 ? "512" : "???")
          + background_symbol_for_tile(tile);
  }
}


void fancy_draw_board(std::ostream &out, const board &b) {
  const int cell_h = 5, cell_w = 12;
  assert(cell_h >= 1 && cell_w >= 4);
  out << "+";
  for (int j = 0; j < 4; ++j) {
    for (int l = 0; l < cell_w; ++l) {
      out << "-";
    }
    out << "+";
  }
  out << "\n";
  for (int i = 0; i < 4; ++i) {
    for (int k = 0; k < (cell_h - 1) / 2; ++k) {
      out << "|";
      for (int j = 0; j < 4; ++j) {
        for (int l = 0; l < cell_w; ++l) {
          out << background_symbol_for_tile(b.f[i][j]);
        }
        out << "|";
      }
      out << "\n";
    }
    out << "|";
    for (int j = 0; j < 4; ++j) {
      for (int l = 0; l < cell_w / 2 - 2; ++l) {
        out << background_symbol_for_tile(b.f[i][j]);
      }
      out << string_for_tile(b.f[i][j]);
      for (int l = 0; l < cell_w  - cell_w / 2 - 2; ++l) {
        out << background_symbol_for_tile(b.f[i][j]);
      }
      out << "|";
    }
    out << "\n";
    for (int k = 0; k < cell_h - 1 - (cell_h - 1) / 2; ++k) {
      out << "|";
      for (int j = 0; j < 4; ++j) {
        for (int l = 0; l < cell_w; ++l) {
          out << background_symbol_for_tile(b.f[i][j]);
        }
        out << "|";
      }
      out << "\n";
    }
    out << "+";
    for (int j = 0; j < 4; ++j) {
      for (int l = 0; l < cell_w; ++l) {
        out << "-";
      }
      out << "+";
    }
    out << "\n";
  }
}

std::ostream& operator<< (std::ostream &out, const board &b) {
  fancy_draw_board(out, b);
  return out;
}
