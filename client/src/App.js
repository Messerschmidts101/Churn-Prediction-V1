
import Nav from './Components/Nav'
import Home from './Pages/Home'
import SingleCustomer from './Pages/SingleCustomer'
import MultipleCustomers from './Pages/MultipleCustomers'
import NotFoundPage from './Pages/NotFoundPage'
import { Routes, Route } from 'react-router-dom'
import './CSS/main.css'
import Footer from './Components/Footer'

function App() {
    
    return (
        <>
            <Nav theme="nav" />
            <main className='main bg-light'>
                <Routes>
                    <Route path='/' element={ <Home />} />
                    <Route path='/home' element={ <Home />} />
                    <Route path='/single_customer' element={ <SingleCustomer />} />
                    <Route path='/multiple_customer' element={ <MultipleCustomers />} />
                    <Route path='/*' element={ <NotFoundPage /> } />
                </Routes>
            </main>
            <Footer />
        </>
    )
}

export default App
