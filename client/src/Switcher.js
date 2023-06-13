import React from 'react'
import Home from './Pages/Home'
import { BrowserRouter, Routes, Route } from 'react-router-dom'

function Switcher() {
    return (
        <BrowserRouter>
            <Routes>
                <Route path='/' exact>
                    <Home />
                </Route>
                <Route path='/home'>
                    <Home />
                </Route>
                <Route path='/predict'></Route>
            </Routes>
        </BrowserRouter>
    )
}

export default Switcher