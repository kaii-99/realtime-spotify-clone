import GroupPlaylist from "../models/groupPlaylist.model.js";

/**
 *  Create Group Playlist
 */
export const createGroupPlaylist = async (req, res) => {
  try {
    const { name, members} = req.body;

    if (!name || !members || members.length === 0) {
      return res.status(400).json({ message: "Missing required fields" });
    }

    const playlist = await GroupPlaylist.create({
      name,
      members,
    });

    res.status(201).json(playlist);
  } catch (error) {
    console.error("Create Group Playlist Error:", error);
    res.status(500).json({ message: "Failed to create group playlist" });
  }
};


/**
 *  Get all playlists for a user
 */
export const getUserGroupPlaylists = async (req, res) => {
  try {
    const { userId } = req.params;

    const playlists = await GroupPlaylist.find({
      members: userId
    }).sort({ createdAt: -1 });

    res.json(playlists);

  } catch (error) {
    console.error(error);
    res.status(500).json({ message: "Failed to fetch group playlists" });
  }
};


/**
 *  Add song to group playlist
 */
export const addSongToGroupPlaylist = async (req, res) => {
  try {
    const { playlistId } = req.params;
    const { songId } = req.body;

    const updated = await GroupPlaylist.findByIdAndUpdate(
      playlistId,
      { $addToSet: { songs: songId } }, // prevents duplicates
      { new: true }
    ).populate("songs");

    if (!updated) {
      return res.status(404).json({ message: "Playlist not found" });
    }

    res.json(updated);
  } catch (error) {
    console.error("Add song error:", error);
    res.status(500).json({ message: "Failed to add song" });
  }
};
