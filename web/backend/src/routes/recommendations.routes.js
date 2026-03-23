import express from "express";
import { 
    getRecommendations, 
    getRecommendations_MoodEnhanced, 
    getRecommendations_MoodEnhanced_DL, 
    getRecommendations_GroupPlaylist, 
    getRecommendations_GroupPlaylist_DL,
    getRecommendations_Hybrid,
    getRecommendations_Hybrid_DL } from "../controller/recommendations.controller.js"; // your controller

const router = express.Router();

router.get("/", getRecommendations);
router.get("/moodenhanced", getRecommendations_MoodEnhanced);
router.get("/moodenhanced_deeplearning", getRecommendations_MoodEnhanced_DL);
router.get("/groupplaylist", getRecommendations_GroupPlaylist);
router.get("/groupplaylist_deeplearning", getRecommendations_GroupPlaylist_DL);
router.get("/hybrid", getRecommendations_Hybrid);
router.get("/hybrid_deeplearning", getRecommendations_Hybrid_DL);

export default router;
