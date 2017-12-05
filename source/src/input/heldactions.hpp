#ifndef HELDACTIONS_HPP
#define HELDACTIONS_HPP

// HEADER-ONLY probably. Needed to avoid a circular dependency between
// input and state

struct HeldActions {
  // 6 dirs
  bool move_right;
  bool move_left;
  bool move_up;
  bool move_down;
  bool move_forw;
  bool move_back;

  // 6 rotations
  bool turn_left;
  bool turn_right;
  bool turn_up;
  bool turn_down;
  bool roll_right;
  bool roll_left;

  HeldActions()
      : move_right(false), move_left(false), move_up(false), move_down(false),
        move_forw(false), move_back(false), turn_left(false), turn_right(false),
        turn_up(false), turn_down(false), roll_right(false), roll_left(false){};
};
  
#endif
