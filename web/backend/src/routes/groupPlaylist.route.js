import express from "express";
import {createGroupPlaylist, getUserGroupPlaylists, addSongToGroupPlaylist, fetchSongGroupPlaylists } from "../controller/groupPlaylist.controller.js";

const router = express.Router();

router.post("/", createGroupPlaylist);
router.get("/:userId", getUserGroupPlaylists);
router.get("/fetchsong/:groupId", fetchSongGroupPlaylists);
router.patch("/:playlistId/add-song", addSongToGroupPlaylist);

export default router;
