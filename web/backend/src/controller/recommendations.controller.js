import axios from "axios";
import { Song } from "../models/song.model.js";

export const getRecommendations = async (req, res) => {
  try {
    const { songIds } = req.query;

    if (!songIds) {
      return res.status(400).json({ message: "songIds query parameter is required" });
    }

    const AI_API_URL = process.env.AI_MODEL_URL || "http://localhost:4000";

    // 🔹 Call Python model API
    const aiResponse = await axios.get(`${AI_API_URL}/recommendations`, {
      params: { 
        songIds,
        city: req.query.city || "Singapore"
      },
    });

    // AI model returns an array of recommended song IDs
    const recommendedIds = aiResponse.data;

    // Fetch full song objects from DB
    const recommendedSongs = await Song.find({ _id: { $in: recommendedIds } });

    res.json(recommendedSongs);
  } catch (error) {
    console.error("Error in getRecommendations:", error);
    res.status(500).json({ message: "Internal server error" });
  }
};

export const getRecommendations_MoodEnhanced = async (req, res) => {
  try {
    const { user_id } = req.query;

    if (!user_id) {
      return res.status(400).json({ message: "user_id query parameter is required" });
    }

    const AI_API_URL = process.env.AI_MODEL_URL || "http://localhost:4000";

    // 🔹 Call Python model API
    const aiResponse = await axios.get(`${AI_API_URL}/recommendations_moodenhanced`, {
      params: { 
        user_id,
        city: req.query.city || "Singapore"
      },
    });

    // AI model returns an array of recommended song IDs
    const recommendedIds_moodenhanced = aiResponse.data.map(item => item.song_id);

    console.log("IDs from AI:", recommendedIds_moodenhanced);
    // Fetch full song objects from DB
    const recommendedSongs_moodenhanced = await Song.find({
      _id: { $in: recommendedIds_moodenhanced }
    });

    res.json(recommendedSongs_moodenhanced);
  } catch (error) {
    console.error("Error in getRecommendations_MoodEnhanced:", error);
    res.status(500).json({ message: "Internal server error" });
  }
};

export const getRecommendations_MoodEnhanced_DL = async (req, res) => {
  try {
    const { user_id } = req.query;

    if (!user_id) {
      return res.status(400).json({ message: "user_id query parameter is required" });
    }

    const AI_API_URL = process.env.AI_MODEL_URL || "http://localhost:4000";

    // 🔹 Call Python model API
    const aiResponse = await axios.get(`${AI_API_URL}/recommendations_moodenhanced_deeplearning`, {
      params: { 
        user_id,
        city: req.query.city || "Singapore"
      },
    });

    // AI model returns an array of recommended song IDs
    const recommendedIds_moodenhanced = aiResponse.data.map(item => item.song_id);

    console.log("IDs from AI:", recommendedIds_moodenhanced);
    // Fetch full song objects from DB
    const recommendedSongs_moodenhanced = await Song.find({
      _id: { $in: recommendedIds_moodenhanced }
    });

    res.json(recommendedSongs_moodenhanced);
  } catch (error) {
    console.error("Error in getRecommendations_MoodEnhanced_DL:", error);
    res.status(500).json({ message: "Internal server error" });
  }
};

export const getRecommendations_GroupPlaylist = async (req, res) => {
  try {
    const { group_id } = req.query;

    if (!group_id) {
      return res.status(400).json({ message: "group_id query parameter is required" });
    }

    const AI_API_URL = process.env.AI_MODEL_URL || "http://localhost:4000";

    // 🔹 Call Python model API
    const aiResponse = await axios.get(`${AI_API_URL}/recommendations_groupplaylist`, {
      params: { 
        group_id
      },
    });

    // AI model returns an array of recommended song IDs
    const recommendedIds_groupplaylist = aiResponse.data.map(item => item.song_id);

    console.log("IDs from AI:", recommendedIds_groupplaylist);
    // Fetch full song objects from DB
    const recommendedSongs_groupplaylist = await Song.find({
      _id: { $in: recommendedIds_groupplaylist }
    });

    res.json(recommendedSongs_groupplaylist);
  } catch (error) {
    console.error("Error in getRecommendations_groupplaylist:", error);
    res.status(500).json({ message: "Internal server error" });
  }
};

export const getRecommendations_GroupPlaylist_DL = async (req, res) => {
  try {
    const { group_id } = req.query;

    if (!group_id) {
      return res.status(400).json({ message: "group_id query parameter is required" });
    }

    const AI_API_URL = process.env.AI_MODEL_URL || "http://localhost:4000";

    // 🔹 Call Python model API
    const aiResponse = await axios.get(`${AI_API_URL}/recommendations_groupplaylist_deeplearning`, {
      params: { 
        group_id
      },
    });

    // AI model returns an array of recommended song IDs
    const recommendedIds_groupplaylist = aiResponse.data.map(item => item.song_id);

    console.log("IDs from AI:", recommendedIds_groupplaylist);
    // Fetch full song objects from DB
    const recommendedSongs_groupplaylist = await Song.find({
      _id: { $in: recommendedIds_groupplaylist }
    });

    res.json(recommendedSongs_groupplaylist);
  } catch (error) {
    console.error("Error in getRecommendations_groupplaylist:", error);
    res.status(500).json({ message: "Internal server error" });
  }
};

export const getRecommendations_Hybrid = async (req, res) => {
  try {
    const { group_id } = req.query;

    if (!group_id) {
      return res.status(400).json({ message: "group_id query parameter is required" });
    }

    const AI_API_URL = process.env.AI_MODEL_URL || "http://localhost:4000";

    // 🔹 Call Python model API
    const aiResponse = await axios.get(`${AI_API_URL}/recommendations_hybrid`, {
      params: { 
        group_id,
        city: req.query.city || "Singapore"
      },
    });

    // AI model returns an array of recommended song IDs
    const recommendedIds_hybrid = aiResponse.data.map(item => item.song_id);

    console.log("IDs from AI:", recommendedIds_hybrid);
    // Fetch full song objects from DB
    const recommendedSongs_hybrid = await Song.find({
      _id: { $in: recommendedIds_hybrid }
    });

    res.json(recommendedSongs_hybrid);
  } catch (error) {
    console.error("Error in getRecommendations_hybrid:", error);
    res.status(500).json({ message: "Internal server error" });
  }
};

export const getRecommendations_Hybrid_DL = async (req, res) => {
  try {
    const { group_id } = req.query;

    if (!group_id) {
      return res.status(400).json({ message: "group_id query parameter is required" });
    }

    const AI_API_URL = process.env.AI_MODEL_URL || "http://localhost:4000";

    // 🔹 Call Python model API
    const aiResponse = await axios.get(`${AI_API_URL}/recommendations_hybrid_deeplearning`, {
      params: { 
        group_id,
        city: req.query.city || "Singapore"
      },
    });

    // AI model returns an array of recommended song IDs
    const recommendedIds_hybrid = aiResponse.data.map(item => item.song_id);

    console.log("IDs from AI:", recommendedIds_hybrid);
    // Fetch full song objects from DB
    const recommendedSongs_hybrid = await Song.find({
      _id: { $in: recommendedIds_hybrid }
    });

    res.json(recommendedSongs_hybrid);
  } catch (error) {
    console.error("Error in getRecommendations_hybrid_DL:", error);
    res.status(500).json({ message: "Internal server error" });
  }
};