
import Nav from './Components/Nav'
import Home from './Pages/Home'
import SingleCustomer from './Pages/SingleCustomer'
import MultipleCustomers from './Pages/MultipleCustomers'
import { Routes, Route } from 'react-router-dom'
import './CSS/main.css'

function App() {


    
    return (
        <main className='main'>
            <Nav theme="nav" />
            <Routes>
                <Route path='/' element={ <Home />} />
                <Route path='/single_customer' element={ <SingleCustomer />} />
                <Route path='/multiple_customer' element={ <MultipleCustomers />} />
            </Routes>
        </main>
    )
}

export default App
