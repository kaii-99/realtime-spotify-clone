import express from "express";
import { getRecommendations } from "../controller/recommendations.controller.js"; // your controller

const router = express.Router();

router.get("/", getRecommendations);

export default router;
