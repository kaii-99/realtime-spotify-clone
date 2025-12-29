import mongoose from "mongoose";

const groupPlaylistSchema = new mongoose.Schema(
  {
    name: {
      type: String,
      required: true,
      trim: true
    },

    members: [
      {
        type: String, // user id
        required: true
      }
    ],

    songs: [
      {
        type: mongoose.Schema.Types.ObjectId,
        ref: "Song",
        default: []
      }
    ]
  },
  { timestamps: true }
);

const GroupPlaylist = mongoose.model("GroupPlaylist", groupPlaylistSchema);
export default GroupPlaylist;
