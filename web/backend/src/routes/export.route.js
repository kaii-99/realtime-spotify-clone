import express from "express";
import { exportListeningData, exportAllSongData, exportMoodData, exportMoodData_Personalized, fetchUserGroupPlaylists } from "../controller/export.controller.js";

const router = express.Router();

router.get("/listenhistory", exportListeningData);
router.get("/allsong", exportAllSongData);
router.get("/listenhistory_mood", exportMoodData);
router.get("/listenhistory_mood/:userId", exportMoodData_Personalized);
router.get("/group_member/:groupId", fetchUserGroupPlaylists);

export default router;
