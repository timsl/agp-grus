#ifndef HELDACTIONS_HPP
#define HELDACTIONS_HPP

// HEADER-ONLY probably. Needed to avoid a circular dependency between
// input and state

struct HeldActions {
  // 6 dirs
  bool move_right = false;
  bool move_left = false;
  bool move_up = false;
  bool move_down = false;
  bool move_forw = false;
  bool move_back = false;

  // 6 rotations
  bool turn_left = false;
  bool turn_right = false;
  bool turn_up = false;
  bool turn_down = false;
  bool roll_right = false;
  bool roll_left = false;

  bool simulation_running = false;

  HeldActions() {}
};

#endif
